

#include <iostream>
#include <string>
#include <vector>
using namespace std;



void recurse(int iteration, string str, int length, vector<string>& nums){
	if (str.size() == length){
	    cout << str << endl;
		nums.push_back(str);
		return;
	}
	int size = str.size();
	int nextNum = str.size() + 1;
	string nextNumString = to_string(nextNum);
	for (int i=0; i<=str.size(); i++){
		string copy = str;
		int pos = size - i;
		copy.insert(pos, nextNumString);
		recurse(iteration + 1, copy, length, nums);
	}
}


int main(){
	int length;
	cin >> length;
	// string str = to_string(length)
	// int iteration;
	vector<string> nums;
	string str = "1";
	recurse(1, str, length, nums);
// 	cout << nums[1727 + 1] << endl;
// 	cout << nums[2232 + 1] << endl;
// 	cout << nums[3777 + 1] << endl;
    // cout << nums[42] << endl;
    // cout << nums[54] << endl;
	return 0;
}


