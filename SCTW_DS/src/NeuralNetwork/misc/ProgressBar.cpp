#include <iostream>
#include <string>

using namespace std;

class ProgressBar {
 private:
  int total;
  int current;
  int bar_length;
  string message;

 public:
  ProgressBar(int total, int bar_length = 20,
              const string &message = "Finished!")
      : total(total), current(0), bar_length(bar_length), message(message) {
    std::cout << "\n\x1b[32m----Starting Training----\x1b[0m\n" << std::endl;
    update(0, 0.0);
  }

  void update(int current, double cost) {
    double fraction = static_cast<double>(current) / total;

    string arrow;
    for (int i = 0; i < static_cast<int>(fraction * bar_length) - 1; ++i) {
      arrow += "-";
    }
    arrow += ">";

    string padding;
    for (int i = 0; i < bar_length - arrow.size(); ++i) {
      padding += " ";
    }

    string ending = current == total ? "\n\n" + message + "\n\n" : "\r";
    string completed = current == total ? "\x1b[32m" : "\x1b[0m";

    cout << completed << "Progress: [" << arrow << padding << "] "
         << fraction * 100 << "%" << " Cost: " << cost << ending;
  }

  void increment(int step = 1, double cost = 0.0) {
    current += step;
    update(current, cost);
  }
};
