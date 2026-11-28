// Bounded producer-consumer queue of slot indices (render/ring.py). A slot is always in exactly one
// place: free list -> producer -> ready FIFO -> consumer -> free list, so its buffer has one owner.
#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>

namespace render {

class SlotQueue {
 public:
  // drop_oldest: a producer that finds no free slot takes back the oldest frame not yet rendered
  // (lowest latency); otherwise it blocks (back-pressure: every frame is rendered).
  SlotQueue(int n, bool drop_oldest) : drop_(drop_oldest) {
    for (int i = 0; i < n; ++i) free_.push_back(i);
  }
  int acquire() {  // producer; -1 once the queue is closed
    std::unique_lock<std::mutex> l(m_);
    cv_free_.wait(l, [&] { return closed_ || !free_.empty() || (drop_ && !ready_.empty()); });
    if (closed_) return -1;
    std::deque<int>& from = free_.empty() ? ready_ : free_;
    if (free_.empty()) ++dropped_;
    int i = from.front();
    from.pop_front();
    return i;
  }
  void publish(int i) {  // producer
    { std::lock_guard<std::mutex> l(m_); ready_.push_back(i); }
    cv_ready_.notify_one();
  }
  // consumer: next ready slot; -1 on timeout, -2 when the producer finished and everything is drained
  int take(int timeout_ms) {
    std::unique_lock<std::mutex> l(m_);
    cv_ready_.wait_for(l, std::chrono::milliseconds(timeout_ms), [&] { return !ready_.empty() || finished_ || closed_; });
    if (ready_.empty()) return finished_ || closed_ ? -2 : -1;
    int i = ready_.front();
    ready_.pop_front();
    return i;
  }
  void release(int i) {  // consumer
    { std::lock_guard<std::mutex> l(m_); free_.push_back(i); }
    cv_free_.notify_one();
  }
  void finish() {  // producer: no more frames
    { std::lock_guard<std::mutex> l(m_); finished_ = true; }
    cv_ready_.notify_all();
  }
  void close() {  // consumer gone (window closed): wake and stop the producer
    { std::lock_guard<std::mutex> l(m_); closed_ = true; }
    cv_free_.notify_all();
    cv_ready_.notify_all();
  }
  long dropped() {
    std::lock_guard<std::mutex> l(m_);
    return dropped_;
  }

 private:
  std::mutex m_;
  std::condition_variable cv_free_, cv_ready_;
  std::deque<int> free_, ready_;
  bool drop_, finished_ = false, closed_ = false;
  long dropped_ = 0;
};

}  // namespace render
