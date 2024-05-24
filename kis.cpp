

// C Program to demonstrate use
// of left shift  operator
#include <stdio.h>
#include <iostream>
using namespace std;
 
// Driver code
int main()
{
    // a = 5(00000101), b = 9(00001001)
    unsigned char a = 0b0101;
 
    // The result is 00001010
    std::cout << int(a) << std::endl;
    return 0;
}
