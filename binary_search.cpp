#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int binarySearch(const vector<int>& arr, int x) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int mid = l + ((r - l) >> 1); // Avoid potential overflow with right shift
        
        if (arr[mid] < x)
            l = mid + 1;
        else if (arr[mid] > x)
            r = mid - 1;
        else
            return mid; // Found
    }
    return -1; // Not found
}

int main() {
    int n, x;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        cin >> arr[i];
    
    sort(arr.begin(), arr.end()); // Ensure array is sorted
    
    cin >> x;
    int result = binarySearch(arr, x);
    if (result == -1)
        cout << "Not found";
    else
        cout << "Found at index " << result;
    return 0;
}
