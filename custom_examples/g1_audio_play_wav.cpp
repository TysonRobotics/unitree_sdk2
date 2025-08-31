#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <atomic>
#include <thread>
#include <cstdint>
#include <algorithm>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/select.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>
#include <unitree/common/time/time_tool.hpp>

// Reuse the simple WAV loader used by the SDK example
#include "wav.hpp"

static const char* GROUP_IP = "239.168.123.161";
static const int PORT = 5555;

static std::string getIPv4ForInterface(const std::string &iface) {
  struct ifaddrs *ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) return "";
  std::string result = "";
  for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr) continue;
    if (ifa->ifa_addr->sa_family != AF_INET) continue;
    if (!ifa->ifa_name) continue;
    if (iface != ifa->ifa_name) continue;
    char host[NI_MAXHOST];
    if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
      result = host;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

class TerminalRawModeGuard {
 public:
  TerminalRawModeGuard() : enabled_(false), orig_flags_(0) {
    if (tcgetattr(STDIN_FILENO, &orig_) == 0) {
      enabled_ = true;
      orig_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
      enterRawNonblock();
    }
  }
  ~TerminalRawModeGuard() {
    if (enabled_) {
      // restore canonical+echo and original flags
      tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
      fcntl(STDIN_FILENO, F_SETFL, orig_flags_);
    }
  }

  void enterLineMode() {
    if (!enabled_) return;
    termios cooked = orig_;
    cooked.c_lflag |= (ICANON | ECHO);
    cooked.c_cc[VMIN] = 1;
    cooked.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &cooked);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
  }

  void enterRawNonblock() {
    if (!enabled_) return;
    termios raw = orig_;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
  }

 private:
  termios orig_{};
  bool enabled_;
  int orig_flags_;
};

struct Recorder {
  std::atomic<bool> isRecording{false};
  std::atomic<bool> stopRequested{false};
  std::thread worker;
  std::string localIp;

  void start(const std::string &outputPath) {
    if (isRecording.load()) return;
    stopRequested.store(false);
    isRecording.store(true);
    worker = std::thread([this, outputPath]() {
      int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
      if (sock < 0) {
        std::cerr << "Failed to create socket for recording" << std::endl;
        isRecording.store(false);
        return;
      }
      int reuse = 1;
      setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));

      sockaddr_in local{};
      local.sin_family = AF_INET;
      local.sin_port = htons(PORT);
      local.sin_addr.s_addr = INADDR_ANY;
      if (bind(sock, (sockaddr*)&local, sizeof(local)) < 0) {
        std::cerr << "bind() failed for recording socket" << std::endl;
        close(sock);
        isRecording.store(false);
        return;
      }

      ip_mreq mreq{};
      inet_pton(AF_INET, GROUP_IP, &mreq.imr_multiaddr);
      mreq.imr_interface.s_addr = inet_addr(localIp.c_str());
      if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        std::cerr << "IP_ADD_MEMBERSHIP failed; localIp=" << localIp << std::endl;
        close(sock);
        isRecording.store(false);
        return;
      }

      std::vector<int16_t> pcm;
      pcm.reserve(16000 * 10); // pre-allocate ~10s
      std::cout << "[rec] Recording... press r to stop" << std::endl;
      while (!stopRequested.load()) {
        char buffer[2048];
        ssize_t len = recvfrom(sock, buffer, sizeof(buffer), MSG_DONTWAIT, nullptr, nullptr);
        if (len > 0) {
          size_t samples = static_cast<size_t>(len) / 2;
          const int16_t *samplesPtr = reinterpret_cast<const int16_t*>(buffer);
          pcm.insert(pcm.end(), samplesPtr, samplesPtr + samples);
        } else {
          // avoid busy spin
          unitree::common::Sleep(1);
        }
      }
      close(sock);

      if (!pcm.empty()) {
        WriteWave(outputPath.c_str(), 16000, pcm.data(), static_cast<int32_t>(pcm.size()), 1);
        std::cout << "[rec] Saved " << outputPath << " (samples=" << pcm.size() << ")" << std::endl;
      } else {
        std::cout << "[rec] No data captured" << std::endl;
      }
      isRecording.store(false);
    });
  }

  void stop() {
    if (!isRecording.load()) return;
    stopRequested.store(true);
    if (worker.joinable()) worker.join();
    stopRequested.store(false);
  }
};

struct Player {
  std::atomic<bool> isPlaying{false};
  std::atomic<bool> stopRequested{false};
  std::thread worker;
  unitree::robot::g1::AudioClient *client{nullptr};

  void start(const std::string &wavPath) {
    if (isPlaying.load() || client == nullptr) return;
    stopRequested.store(false);
    isPlaying.store(true);
    worker = std::thread([this, wavPath]() {
      int32_t rate = -1; int8_t ch = 0; bool ok = false;
      std::vector<uint8_t> pcm = ReadWave(wavPath.c_str(), &rate, &ch, &ok);
      if (!ok || rate != 16000 || ch != 1) {
        std::cerr << "[play] Invalid WAV (need 16kHz mono): " << wavPath << std::endl;
        isPlaying.store(false);
        return;
      }
      // 16kHz mono 16-bit => 32000 bytes/sec. Use 1s chunks to match realtime.
      const size_t CHUNK = 32000;
      std::string stream_id = std::to_string(unitree::common::GetCurrentTimeMillisecond());
      size_t off = 0, total = pcm.size();
      std::cout << "[play] Playing... press p to stop" << std::endl;
      while (off < total && !stopRequested.load()) {
        size_t cur = std::min(CHUNK, total - off);
        std::vector<uint8_t> chunk(pcm.begin() + off, pcm.begin() + off + cur);
        client->PlayStream("custom", stream_id, chunk);
        unitree::common::Sleep(1);
        off += cur;
      }
      client->PlayStop(stream_id);
      isPlaying.store(false);
      std::cout << "[play] Stopped" << std::endl;
    });
  }

  void stop() {
    if (!isPlaying.load()) return;
    stopRequested.store(true);
    if (worker.joinable()) worker.join();
    stopRequested.store(false);
  }
};

// Simple TCP speaker bridge: receive raw 16k mono 16-bit PCM bytes and stream to robot
struct SpeakerBridge {
  std::thread server;
  std::atomic<bool> running{false};
  unitree::robot::g1::AudioClient *client{nullptr};

  void start(uint16_t port = 5002) {
    if (running.load() || client == nullptr) return;
    running.store(true);
    server = std::thread([this, port]() {
      int srv = ::socket(AF_INET, SOCK_STREAM, 0);
      if (srv < 0) { std::cerr << "[bridge] socket() failed" << std::endl; running.store(false); return; }
      int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = htons(port);
      if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { std::cerr << "[bridge] bind() failed" << std::endl; close(srv); running.store(false); return; }
      if (listen(srv, 1) < 0) { std::cerr << "[bridge] listen() failed" << std::endl; close(srv); running.store(false); return; }
      std::cout << "[bridge] Listening on 127.0.0.1:" << port << " for PCM (16k mono 16-bit)" << std::endl;
      const size_t CHUNK = 32000; // 1s of 16kHz mono s16le
      while (running.load()) {
        sockaddr_in cli{}; socklen_t cl = sizeof(cli);
        int fd = accept(srv, (sockaddr*)&cli, &cl);
        if (fd < 0) { if (!running.load()) break; continue; }
        std::string stream_id = std::to_string(unitree::common::GetCurrentTimeMillisecond());
        std::vector<uint8_t> buf; buf.reserve(CHUNK);
        std::cout << "[bridge] Client connected" << std::endl;
        while (running.load()) {
          uint8_t tmp[4096]; ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
          if (n <= 0) break;
          buf.insert(buf.end(), tmp, tmp + n);
          while (buf.size() >= CHUNK) {
            std::vector<uint8_t> out(buf.begin(), buf.begin() + CHUNK);
            client->PlayStream("bridge", stream_id, out);
            buf.erase(buf.begin(), buf.begin() + CHUNK);
            // pace approximately 1s per chunk
            unitree::common::Sleep(1);
          }
        }
        // flush remainder
        if (!buf.empty()) {
          client->PlayStream("bridge", stream_id, buf);
          buf.clear();
        }
        client->PlayStop(stream_id);
        close(fd);
        std::cout << "[bridge] Client disconnected" << std::endl;
      }
      close(srv);
    });
  }

  void stop() {
    if (!running.load()) return;
    running.store(false);
    // connect to self to unblock accept
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) { sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(5002); connect(fd, (sockaddr*)&a, sizeof(a)); close(fd); }
    if (server.joinable()) server.join();
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: g1_audio_play_wav <network_interface> [wav_path]\n";
    return 1;
  }

  const std::string iface = argv[1];
  const std::string default_wav_path = (argc >= 3) ? argv[2] : std::string("/home/pim/Desktop/unitree_sdk2/build/record.wav");

  unitree::robot::ChannelFactory::Instance()->Init(0, iface);

  unitree::robot::g1::AudioClient client;
  client.Init();
  client.SetTimeout(10.0f);

  std::string localIp = getIPv4ForInterface(iface);
  if (localIp.empty()) {
    std::cerr << "Failed to find IPv4 for interface: " << iface << std::endl;
    return 2;
  }

  Recorder recorder;
  recorder.localIp = localIp;
  Player player;
  player.client = &client;
  SpeakerBridge bridge; bridge.client = &client; bridge.start(5002);
  int ttsVoiceId = 1; // 0: Chinese, 1: English (per SDK example)

  std::cout << "Controls: r=start/stop record, p=start/stop play, t=tts, c=CN voice, e=EN voice, 0/1=voice id, +=/= vol up, -=vol down, v=set vol, g=get vol, q=quit\n";
  std::cout << "Default play file: " << default_wav_path << "\n";

  TerminalRawModeGuard tty;
  bool running = true;
  while (running) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    timeval tv{0, 200000}; // 200ms
    int rv = select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv);
    if (rv > 0 && FD_ISSET(STDIN_FILENO, &readfds)) {
      char ch;
      ssize_t n = read(STDIN_FILENO, &ch, 1);
      if (n == 1) {
        if (ch == 'q' || ch == 'Q') {
          running = false;
        } else if (ch == 'r' || ch == 'R') {
          if (!recorder.isRecording.load()) recorder.start("/home/pim/Desktop/unitree_sdk2/build/record.wav");
          else recorder.stop();
        } else if (ch == 'p' || ch == 'P') {
          if (!player.isPlaying.load()) player.start(default_wav_path);
          else player.stop();
        } else if (ch == 't' || ch == 'T') {
          // temporarily switch to line mode to read a full line
          tty.enterLineMode();
          std::cout << "Enter text to speak: " << std::flush;
          std::string line;
          std::getline(std::cin, line);
          tty.enterRawNonblock();
          if (!line.empty()) {
            int32_t ret = client.TtsMaker(line.c_str(), ttsVoiceId);
            std::cout << "[tts] voice_id=" << ttsVoiceId << " ret=" << ret << std::endl;
          }
        } else if (ch == 'c' || ch == 'C') {
          ttsVoiceId = 0; std::cout << "[tts] Voice set to Chinese (0)" << std::endl;
        } else if (ch == 'e' || ch == 'E') {
          ttsVoiceId = 1; std::cout << "[tts] Voice set to English (1)" << std::endl;
        } else if (ch == '0') {
          ttsVoiceId = 0; std::cout << "[tts] Voice id = 0" << std::endl;
        } else if (ch == '1') {
          ttsVoiceId = 1; std::cout << "[tts] Voice id = 1" << std::endl;
        } else if (ch == '+' || ch == '=') {
          uint8_t vol = 0; int32_t gr = client.GetVolume(vol);
          std::cout << "[vol] get ret=" << gr << " val=" << static_cast<int>(vol) << std::endl;
          if (gr == 0) {
            int nv = std::min(100, static_cast<int>(vol) + 10);
            int32_t sr = client.SetVolume(static_cast<uint8_t>(nv));
            std::cout << "[vol] set ret=" << sr << " -> " << nv << std::endl;
          }
        } else if (ch == '-' ) {
          uint8_t vol = 0; int32_t gr = client.GetVolume(vol);
          std::cout << "[vol] get ret=" << gr << " val=" << static_cast<int>(vol) << std::endl;
          if (gr == 0) {
            int nv = static_cast<int>(vol) - 10; if (nv < 0) nv = 0;
            int32_t sr = client.SetVolume(static_cast<uint8_t>(nv));
            std::cout << "[vol] set ret=" << sr << " -> " << nv << std::endl;
          }
        } else if (ch == 'v' || ch == 'V') {
          tty.enterLineMode();
          std::cout << "Enter volume (0-100): " << std::flush;
          std::string line;
          std::getline(std::cin, line);
          tty.enterRawNonblock();
          try {
            int nv = std::stoi(line);
            if (nv < 0) nv = 0; if (nv > 100) nv = 100;
            int32_t sr = client.SetVolume(static_cast<uint8_t>(nv));
            std::cout << "[vol] set ret=" << sr << " -> " << nv << std::endl;
            uint8_t vol = 0; int32_t gr = client.GetVolume(vol);
            std::cout << "[vol] get ret=" << gr << " now=" << static_cast<int>(vol) << std::endl;
          } catch (...) { std::cout << "[vol] invalid" << std::endl; }
        } else if (ch == 'g' || ch == 'G') {
          uint8_t vol = 0; int32_t gr = client.GetVolume(vol);
          std::cout << "[vol] get ret=" << gr << " val=" << static_cast<int>(vol) << std::endl;
        }
      }
    }
    // Join finished background threads to avoid std::terminate on destruction
    if (!player.isPlaying.load() && player.worker.joinable()) {
      player.worker.join();
    }
    if (!recorder.isRecording.load() && recorder.worker.joinable()) {
      recorder.worker.join();
    }
    // avoid busy loop
    unitree::common::Sleep(1);
  }

  // ensure threads finish
  recorder.stop();
  player.stop();
  bridge.stop();
  return 0;
}


