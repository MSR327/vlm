#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        int n;
        scanf("%d", &n);
        vector<int> a(n);
        for (int i = 0; i < n; i++) scanf("%d", &a[i]);

        // Collect positions of fixed 1s, and first/last -1
        vector<int> ones;
        int firstNeg = -1, lastNeg = -1;
        for (int i = 0; i < n; i++) {
            if (a[i] == 1) ones.push_back(i);
            if (a[i] == -1) {
                if (firstNeg == -1) firstNeg = i;
                lastNeg = i;
            }
        }

        int bestScore = 0, bestL = -1, bestR = -1;

        auto tryUpdate = [&](int l, int r) {
            int score = r - l + 1;
            if (score > bestScore) {
                bestScore = score;
                bestL = l;
                bestR = r;
            }
        };

        if (ones.empty()) {
            // No fixed 1s — use leftmost and rightmost -1
            if (firstNeg != -1) {
                tryUpdate(firstNeg, lastNeg); // works even if firstNeg == lastNeg (score 1)
            }
        } else {
            // A single fixed 1 already gives score 1
            tryUpdate(ones[0], ones[0]);

            // Between every pair of consecutive fixed 1s
            for (int i = 0; i + 1 < (int)ones.size(); i++) {
                tryUpdate(ones[i], ones[i + 1]);
            }

            // Extend left: leftmost -1 before the first fixed 1
            if (firstNeg != -1 && firstNeg < ones[0]) {
                tryUpdate(firstNeg, ones[0]);
            }

            // Extend right: rightmost -1 after the last fixed 1
            if (lastNeg != -1 && lastNeg > ones.back()) {
                tryUpdate(ones.back(), lastNeg);
            }
        }

        // Reconstruct: set chosen endpoints to 1, all other -1s to 0
        for (int i = 0; i < n; i++) {
            if (a[i] == -1) {
                a[i] = (i == bestL || i == bestR) ? 1 : 0;
            }
        }

        for (int i = 0; i < n; i++) {
            printf("%d%c", a[i], " \n"[i == n - 1]);
        }
    }
    return 0;
}
