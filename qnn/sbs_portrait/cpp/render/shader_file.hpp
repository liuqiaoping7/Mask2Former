#pragma once
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace render {
inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
}  // namespace render
