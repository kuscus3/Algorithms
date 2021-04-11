

# Algorithms

| Data Structures                               | Algorithms                                  |
| --------------------------------------------- | ------------------------------------------- |
| [Array](#Array)                               | [Recursion](#Recursion)                     |
| [String](#String)                             | [BFS, DFS](#BFS, DFS)                       |
| [Linked List](#Linked List)                   | [Dynamic Programming](#Dynamic Programming) |
| [Stack](#Stack)                               | [Backtracking](#Backtracking)               |
| [Queue](#Queue)                               | [Greedy](#Greedy)                           |
| [Hash Table](#Hash Table)                     | [Mathematics](#Mathematics)                 |
| [Trees - Binary, Binary Search, Heap](#Trees) | [Bit Masking](#Bit Masking)                 |
| [Design](#Design)                             | Others - UF, Graphs, Two pointers           |

## Dynamic Programming

1. create recurrence relation like f(n) = f(n-1) + f(n-2)

   * general case
     * specifically define what dp term means ex) dp[i] = the length of LIS on element i
     * express answer in dp terms ex) ans = max(dp[i])
   * base case: where recursive function/iterative loop ends

2. top-down vs bottom-up: use own template and apply the recurrence relation

3. represent in graph: nodes=dp terms, edges=relations

   * DP solution assures DAG

   

Categories

* 0/1 Knapsack
* Unbounded Knapsack
* Shortest Path (eg: Unique Paths I/II)
* Fibonacci Sequence (eg: House Thief, Jump Game)
* Longest Common Substring/Subsequeunce



From [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns/865611)

#### Longest Increasing Subsequence variants

##### 300. Longest Increasing Subsequence

````python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]*len(nums)
        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        
        return max(dp)
````

* dp[i] : LIS from 0 to ith element
* for each ith element, compare 0 to i-1th element and if it's larger update dp values

$$
dp[i]=max(dp[j])+1,\ ∀0≤j<i \ and \ num[i] > num[j]
$$

* answer: maximum value in dp list

##### 368. Largest Divisible Subset

* sort list so that nums[i] % nums[j] == 0 can be checked only once
* dp[i]: [LIS, previous index of LIS]
  * answer: from the index of maximum LIS, traverse previous indexes, finally reverse as traversing in opposite direction

````python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        
        dp = [[1, i] for i in range(len(nums))]
        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if dp[j][0]+1 > dp[i][0]:
                        dp[i] = [dp[j][0]+1,j]
        
        res = list()
        idx = dp.index(max(dp))
        while dp[idx][0] != 1:
            res.append(nums[idx])
            idx = dp[idx][1]
        res += [nums[idx]]
        res.reverse()
        return res
````

##### 646. Maximum Length of Pair Chain

* dp[i] = Longest chain from 0 to ith element

````python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort()
        dp=[1]*len(pairs)
        
        for i in range(1, len(pairs)):
            for j in range(i):
                if pairs[j][1]<pairs[i][0]:
                    dp[i]=max(dp[i],dp[j]+1)
        return max(dp)
````

##### 673. Number of Longest Increasing Subsequence

* dp: [LIS length, Number of cases] initialize
  * i starts from 1 in loop, so [1,1] for dp[0]
* solution: dp중 LIS length가 maximum인것을 fetch해와서 return total number of cases
* Approach
  * if previous elements(j) are smaller than ith element, get LIS length for j and add number of cases
    * if jth LIS length is smaller than previously fetched LIS length, do nothing (no need to update)
    * if ith LIS length equals jth LIS length, cumulatively add number of cases
  * LIS length is initialized to 0 at the beginning because [[1,1],[1,1]] make [[1,1],[2,2]] which is misleading

````python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        dp=[[0, 1] for _ in range(len(nums))]
        dp[0]=[1,1]
        
        for i in range(1, len(nums)):
            tmp = list()
            for j in range(i):
                if nums[i]>nums[j]:
                    if dp[i][0] < dp[j][0]:
                        dp[i][0] = dp[j][0]
                        dp[i][1] = dp[j][1]
                    elif dp[i][0] == dp[j][0] and dp[i][0]:
                        dp[i][1] += dp[j][1]
            dp[i][0]+=1

        maxdp = max(dp)[0]
        lisidx = [j[1] for i,j in enumerate(dp) if j[0]==maxdp]
        return sum(lisidx)
````

##### 740. Delete and Earn

* dp bottom-up approach
  * dp[i] = max(dp[i-1], dp[i-2]+nums[i])
* totalnums는 [2,2,3,3,3] => [0,0,4,9]로 변환 dp로 활용하기 위해  idx값== nums의 값

````python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        nums.sort()
        dp = [0]*(max(nums)+1)
        totalnums = [0]*(max(nums)+1)
        for i in range(len(nums)):
            totalnums[nums[i]] += nums[i]
        print(totalnums)
        
        dp[0] = totalnums[0]
        dp[1] = max(totalnums[0], totalnums[1])
        for i in range(2, len(totalnums)):
            dp[i] = max(dp[i-1], dp[i-2]+totalnums[i])
        
        return max(dp[-1], dp[-2])
````

##### 1048. Longest String Chain

```python 
words.sort(key=len) # list를 string의 length로 sort 하는법: reverse=True가능
```

* 주의할점: xb -> xcb는 sequence가 되지만 xb -> bcx이런건 안됨, 그러므로 string의 순서도 신경써야함\
* predecessor를 판별하는 function만들기가 challenging함: slicing 활용
  * 새 알파벳이 1) 앞뒤에 붙을경우, 2) 중간에 붙을경우 구분해서 파악: 2)의 경우 looping을 통해 알아냄

````python
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        def isPredecessor(s1, s2):
            len1, len2 = len(s1), len(s2)
            if len2-len1 != 1: return False
            if s2[1:]==s1 or s2[:-1]==s1: return True
            for i in range(1, len1):
                if s2[:i]+s2[i+1:]==s1: return True
            return False
        
        words.sort(key=len)
        dp = [1]*len(words)
        
        for i in range(1, len(words)):
            for j in range(i):
                if isPredecessor(words[j], words[i]):
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
````

* 결과가 느리지만, improved version은 나중에 해보는걸로....

##### 354. Russian Doll Envelopes

````python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        print(envelopes)
        dp=[1]*len(envelopes)
        
        for i in range(1, len(envelopes)):
            for j in range(i):
                if envelopes[i][1] > envelopes[j][1]:
                    dp[i] = max(dp[i], dp[j]+1)
        
        return max(dp)
````

* Key to this problem was on sorting envelopes list
  * envelopes.sort() than LIS on both dimensions give TLE
  * sorting the list in ascending order in first dim, descending order in second dim
    * LIS: only compare second dimension since first dim has already ordered
  * example: [[6,7],[6,4]] &#8594; [[6,4],[6,7]] gives LIS=2, which is wrong, reverse order assumes picking only 1 element for equal-width envelopes

#### Partition Subset Sum

##### 416. Partition Equal Subset Sum

* tree+dfs: [1,5,11,5] &#8594; [1] [5,11,5] 같이 하나씩 element를 빼면서 part에 추가와 추가되지 않은 경우를 모두 계산
  * part의 sum이 total의 반이면 partition subset이 완성: True 변환

````python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total%2 != 0:
            return False
        
        def dfs(part, rem):
            if sum(part)*2 == total:
                return True
            for i in range(len(rem)):
                part.append(rem[i])
                tmp = rem[:i]+rem[i+1:]
                if dfs(part, tmp): return True
                part.pop()
                if dfs(part, tmp): return True
                
        
        return dfs([], nums)
````

* TLE: because many of the part list is overlapping in the tree &#8594; where dp is required!!

#### Longest Common Subsequence Variant

##### 1143, Longest Common Subsequence

* lcs("aab", "azb") &#8594; 1 + lcs("aa","ax") [match] / lcs("aac","axb") &#8594; max(lcs("aac","ax"), lcs("aa","axb")) [not match]
  * apply this idea to memoization table

````python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for _ in range(len(text2)+1)] for _ in range(len(text1)+1)]
        
        for i in range(len(text1)+1):
            for j in range(len(text2)+1):
                if i==0 or j==0:
                    continue
                elif text1[i-1] == text2[j-1]:
                    dp[i][j] = 1+dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[len(text1)][len(text2)]
````

* match: both texts' previous word lcs + 1, not match: max(i-1,j and i,j-1)
* initialize dp size (n+1)*(m+1) because we need to consider blank string case(base case=0)

##### 712. Minimum ASCII Delete Sum for Two Strings

* definition of dp: cost(by ASCII value) from one string to another

|        | \0                 | **e**        | **a**         | **t**              |
| ------ | ------------------ | ------------ | ------------- | ------------------ |
| **\0** | 0                  | ord(e) = 101 | ord(a)+ord(e) | ord(t)+dp\[i][j-1] |
| **s**  | ord(s)=115         |              |               |                    |
| **e**  | ord(e)+ord(s)      |              |               |                    |
| **a**  | ord(a)+dp\[i-1][j] |              |               |                    |

* dp\[i][j] = up, left, diag dp value에서 서로 상응하는 ord(s1[i-1]) or ord(s2[j-1]) 더한값중 최솟값
  * but if s1[i-1]==s2[j-1], there is not addition from diag dp value (as lcs("abc", "xyc") ==lcs("ab","xy")+1)

````python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        dp = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
    
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                v1, v2 = ord(s1[i-1]), ord(s2[j-1])
                if i==0 and j==0:
                    continue
                elif i==0:
                    dp[i][j] = dp[i][j-1] + v2
                elif j==0:
                    dp[i][j] = dp[i-1][j] + v1
                elif s1[i-1]==s2[j-1]:
                    dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]+v1),dp[i][j-1]+v2)
                else:
                     dp[i][j] = min(min(dp[i-1][j-1]+v1+v2, dp[i-1][j]+v1),dp[i][j-1]+v2)
        return dp[len(s1)][len(s2)]
````

##### 72. Edit Distance

* Like LCS question, it uses (m+1)*(n+1) size dp, but there are some differences

  * Initialize dp[0]\[] or dp[]\[0] as i, not 0
  * Found out this is [Levenshtein Algorithm][https://www.cuelogic.com/blog/the-levenshtein-algorithm] problem

  <img src="C:\Users\82103\Downloads\Maths.jpg.webp" alt="Maths.jpg" style="zoom: 67%;" />

  * get minimum value from left(i, j-1), up(i-1, j), diag(i-1, j-1) of dp matrix

````python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0 if j!=0 else i for j in range(len(word2)+1)] for i in range(len(word1)+1)]
        dp[0] = list(range(len(word2)+1))
        
        for i in range(1,len(word1)+1):
            for j in range(1,len(word2)+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]+1),dp[i][j-1]+1)
                else:
                    dp[i][j] = min(min(dp[i-1][j-1]+1, dp[i-1][j]+1),dp[i][j-1]+1)
        
        return dp[len(word1)][len(word2)]
````

##### 115. Distinct Subsequences

* dp = # of distinct subsequences from s1[i:] and s2[j:], this question loops in reverse direction range(,,-1)
* only s2 moves into next character if there is a match with s1
  * If not match, move to next character s1 dp\[i][j] = dp\[i+1][j]
  * If match, consider both cases where s2 moves into next or not dp\[i][j] = dp\[i+1][j]+dp\[i+1][j+1]

````python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp = [[0 for _ in range(len(t)+1)] for _ in range(len(s)+1)]
        
        for i in range(len(s),-1,-1):
            for j in range(len(t),-1,-1):
                if j==len(t):
                    dp[i][j] = 1
                elif i==len(s):
                    continue
                elif s[i]!=t[j]:
                    dp[i][j] = dp[i+1][j]
                elif s[i]==t[j]:
                    dp[i][j] = dp[i+1][j]+ dp[i+1][j+1]
        
        return dp[0][0]
````

#### **Palindrome**

````python
class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        
        for i in range(len(s)):
            for j in range(i+1,len(s)+1):
                substr = s[i:j]
                print(substr, i, j)
                if substr==substr[::-1]:
                    count += 1
        return count
````

* without dp: O(N^3) solution

````python
class Solution:
    def countSubstrings(self, s: str) -> int:
        dp=[[False for _ in range(len(s))] for _ in range(len(s))]
        res = 0
        
        for i in range(len(s)-1, -1,-1):
            for j in range(i,len(s)):
                if s[i]==s[j] and (j-i<2 or dp[i+1][j-1]):
                    res += 1
                    dp[i][j] = True
        return res
````

* with len(s)*len(s) dp list
  * dp\[i][j]: whether s[i:j+1] is palindrome
    * subproblem: if s[i:j+1] is palindrome, s[i+1:j] is also palindrome
    * base: if i=j ("a") or i+1=j and s[i]=s[j] ("aa") is palindrome

##### 132. Palindrome Partitioning II

````python
class Solution:
    def minCut(self, s: str) -> int:
        if len(s)==2 and s[0]==s[1]:
            return 0
        
        dp = [[0 for _ in range(len(s))] for _ in range(len(s))]
        
        for i in range(len(s)-1,-1,-1):
            for j in range(i, len(s)):
                if s[i]==s[j] and (j-i<2 or dp[i+1][j-1]):
                    dp[i][j] = 1
        
        self.splitmin = 99999999
        def rec(row, split):
            subpalin = [i for i, val in enumerate(dp[row]) if val == 1]
            for idx in subpalin:
                if idx==len(s)-1:
                    self.splitmin = min(self.splitmin, split)
                    break
                rec(idx+1, split+1)
            return
        
        rec(0, 0)
        return self.splitmin
````

* TLE solution: recursive approach from 2d list is very expensive approach

#### Coin Change

##### 322. Coin Change

````python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [99999999 for _ in range(amount+1)]
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount+1):
                if dp[i-coin] != 99999999:
                    dp[i] = min(dp[i-coin]+1, dp[i])
        return dp[amount] if dp[amount]!=99999999 else -1 
````

* coins = [1,2], amount = 5

| 0    | 1    | 2    | 3    | 4    | 5    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | MAX  | MAX  | MAX  | MAX  | MAX  |

* for coin=1

| 0    | 1(start)                                         | 2                                               | 3    | 4    | 5(end) |
| ---- | ------------------------------------------------ | ----------------------------------------------- | ---- | ---- | ------ |
| 0    | 1<br />min(dp[1-1]+1, dp[1]) = min(0+1, MAX) = 1 | 2<br />min(dp[2-1]+1, dp[2]) = min(1+1,MAX) = 2 | 3    | 4    | 5      |

* for coin=2

| 0    | 1    | 2(start)                                     | 3                                            | 4    | 5(end) |
| ---- | ---- | -------------------------------------------- | -------------------------------------------- | ---- | ------ |
| 0    | 1    | 1<br />min(dp[2-2]+1,dp[2]) = min(0+1,2) = 1 | 2<br />min(dp[3-2]+1,dp[3]) = min(1+1,3) = 2 | 2    | 3      |

##### 518. Coin Change 2

````python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0 for _ in range(amount+1)]
        dp[0] = 1
        
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] += dp[i-coin]
        return dp[-1]
````

##### 377. Combination Sum IV

* dp: # of combinations for target i
  * caution: the looping has changed since it need to consider all num in nums for each dp iteration

````python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        
        for i in range(1,target+1):
            for num in nums:
                if i-num >=0:
                    dp[i] += dp[i-num]
        return dp[-1]
````

* I think this can be solved using pizza cut problem (if nums are ascending order range)

##### 279. Perfect Squares

* dp memoization solution

````python
class Solution:
    def numSquares(self, n: int) -> int:
        squares = [num*num for num in range(int(sqrt(n))+1)]
        dp = [999999999 for _ in range(n+1)]
        dp[0] = 0
        
        for square in squares:
            for i in range(square,n+1):
                if dp[i-square] != 999999999:
                    dp[i] = min(dp[i-square]+1, dp[i])
        return dp[-1]
````

* BFS solution

````python
class Solution:
    def numSquares(self, n: int) -> int:
        def is_square(n):
            return n**0.5 == int(n**0.5)
        queue = collections.deque([(n, 1)])
        while queue:
            if is_square(queue[0][0]) or queue[0][0]==1:
                return queue[0][1]
            num, level = queue.popleft()
            i=1
            while i*i<num:
                queue.append((num-i*i,level+1))
                i+=1
````

##### 983. Minimum Cost For Tickets

* dp[i] = min(dp[i+1]+cost(1), dp[i+7]+cost(7), dp[i+30]+cost(30)) where i is cost from day i the 365

#### **Matrix multiplication**

##### 1039. Minimum Score Triangulation of Polygon

* Recursive solution

<img src="C:\Users\82103\Desktop\Github\Algorithms\img\1039.PNG" alt="1039" style="zoom: 67%;" />

* poly(0,6) = triangle(0,3,6) + poly(0,3) + poly(3,6)
  * for k in range(1,n): min(0,n) = 0\*k\*n + min(0,k) + min(k,n)

````python
#Recursive solution: TLE
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        def rec(left, right):
            if right-left <2:
                return 0
            minimum = 9e10
            for k in range(left+1, right):
                minimum = min(minimum, values[left]*values[right]*values[k]
                              +rec(left,k)+rec(k,right))
            return minimum
        return rec(0,len(values)-1)
````

````python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        @lru_cache(None) #Memoization by lru_cache
        def rec(left, right):
            if right-left <2:
                return 0
            minimum = 9e10
            for k in range(left+1, right):
                minimum = min(minimum, values[left]*values[right]*values[k]
                              +rec(left,k)+rec(k,right))
            return minimum
        return rec(0,len(values)-1)
````

````python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        dp = [[0]*n for i in range(n)]
        for l in range(2, n):
            for left in range(0, n - l):
                right = left + l
                dp[left][right] = float("Inf")
                for k in range(left + 1, right):
                    dp[left][right] = min(dp[left][right], dp[left][k] + dp[k][right] + values[left]*values[right]*values[k])
        return dp[0][-1]
````

* dp solution: dp\[i][j] = minimum score from point i to point j

  * if i and j distance lower than 3, cannot make triange: dp\[i][j]=0 for j-i<3
  * for k in between i and j, dp\[i][j] = dp\[i][k] + dp\[k][j] + traingle(i,k,j)

  <img src="C:\Users\82103\Desktop\Github\Algorithms\img\1039dp.PNG" alt="1039dp" style="zoom: 67%;" />

##### 312. Burst Balloons

* Recursion: rec(nums)  = nums[k-1]*nums[k]\*nums[k+1] + rec(nums[:k]+nums[k+1:]) to burst balloon k

````python
#Recursion: TLE solution
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1]+nums+[1]
        def rec(arr):
            if len(arr)<3:
                return 0
            maxval = -1
            for k in range(1, len(arr)-1):
                maxval = max(maxval, arr[k-1]*arr[k]*arr[k+1]+
                             rec(arr[:k]+arr[k+1:]))
            return maxval
        return rec(nums)
````

````python
#Top-down approach dp:still TLE!
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1]+nums+[1]
        dp = dict()
        
        def rec(arr):
            if len(arr)<3:
                return 0
            if tuple(arr) in dp:
                return dp[tuple(arr)]
            maxval = -1
            for k in range(1, len(arr)-1):
                maxval = max(maxval, arr[k-1]*arr[k]*arr[k+1]+
                             rec(arr[:k]+arr[k+1:]))
            dp[tuple(arr)] = maxval
            return maxval
        return rec(nums)
````

* 새롭게 알게된것: list cannot be key in dictionary since mutable+not-hashable, use tuple instead

````python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1]+nums+[1]
        dp = [[0 for _ in range(len(nums))] for _ in range(len(nums))]
        
        for l in range(2, len(nums)):
            for left in range(0, len(nums)-l):
                right = left+l
                for k in range(left+1, right):
                    dp[left][right] = max(dp[left][right],
                                          dp[left][k]+dp[k][right]+
                                          nums[left]*nums[right]*nums[k])
        return dp[0][-1]
````

* for example left=0, right=4, k=2 &#8594; nums[1:3] that burst balloons
  * dp\[left][right] = dp\[left][k] +dp\[k][right] + burst(k)

##### 1130. Minimum Cost Tree From Leaf Values

````python
#TLE solution - recursive approach
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        def rec(arr):
            if len(arr)==1:
                return 0
            minval = 10e9
            for k in range(1,len(arr)):
                # print(arr[:k-1]+[max(arr[k],arr[k-1])]+arr[k+1:])
                minval = min(minval, arr[k]*arr[k-1]+
                             rec(arr[:k-1]+[max(arr[k],arr[k-1])]+arr[k+1:]))
            return minval
        
        return rec(arr)
````

#### Matrix/2D array

##### 1314. Matrix Block Sum

````python
#Simple iterative solution: TLE
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        rlen = len(mat)
        clen = len(mat[0])
        ans = [[0 for _ in range(clen)] for _ in range(rlen)]
        
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                for r in range(max(0,i-K),min(rlen-1,i+K)+1):
                    for c in range(max(0,j-K),min(clen-1,j+K)+1):
                        ans[i][j] += mat[r][c]

        return ans
````

````python
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        rlen = len(mat)
        clen = len(mat[0])
        summat = [[0 for _ in range(clen+1)] for _ in range(rlen+1)]
        for i in range(rlen):
            for j in range(clen):
                summat[i+1][j+1] = summat[i][j+1] + summat[i+1][j] - summat[i][j]+ mat[i][j]
        
        
        ans = [[0 for _ in range(clen)] for _ in range(rlen)]
        for i in range(rlen):
            for j in range(clen):
                rl, cl, rr, cr = max(i-K,0), max(j-K,0), min(i+K+1,rlen), min(j+K+1,clen)
                ans[i][j] = summat[rr][cr]-summat[rr][cl]-summat[rl][cr]+summat[rl][cl]
        
        return ans
````

<img src="C:\Users\82103\Desktop\Github\Algorithms\img\1314a.PNG" alt="1314a" style="zoom:50%;" />

<img src="C:\Users\82103\Desktop\Github\Algorithms\img\1314b.PNG" alt="1314b" style="zoom:50%;" />

##### 304. Range Sum Query 2D - Immutable

````python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.mat = matrix
        rlen = len(self.mat)
        clen = len(self.mat[0])
        self.rangesum = [[0 for _ in range(clen+1)] for _ in range(rlen+1)]
        for i in range(rlen):
            for j in range(clen):
                self.rangesum[i+1][j+1] = self.rangesum[i][j+1] + self.rangesum[i+1][j] - self.rangesum[i][j] + self.mat[i][j]
        print(self.rangesum)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.rangesum[row2+1][col2+1] - self.rangesum[row1][col2+1] - self.rangesum[row2+1][col1] + self.rangesum[row1][col1]
# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
````

##### 120. Triangle

````python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = [[0 for _ in range(i+1)] for i in range(len(triangle))]
        
        dp[0][0] = triangle[0][0]
        
        for i in range(1,len(triangle)):
            for j in range(i+1):
                if j==0: 
                    dp[i][j] = triangle[i][j] + dp[i-1][j]
                elif j==i:
                    dp[i][j] = triangle[i][j] + dp[i-1][j-1]
                else:
                    dp[i][j] = triangle[i][j] + min(dp[i-1][j], dp[i-1][j-1])
        return min(dp[-1])
        
````

##### 221. Maximal Square

````python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rlen,clen = len(matrix),len(matrix[0])
        summat = [[0 for _ in range(clen+1)] for _ in range(rlen+1)]
        for i in range(rlen):
            for j in range(clen):
                summat[i+1][j+1] = summat[i][j+1]+summat[i+1][j]-summat[i][j]+int(matrix[i][j])
        print(summat)
    
        for l in range(max(rlen,clen)-1,-1,-1):
            for r in range(0,rlen-l):
                for c in range(0,clen-l):
                    r1,c1,r2,c2 = r,c,r+l,c+l
                    squaresum = summat[r2+1][c2+1] - summat[r1][c2+1] - summat[r2+1][c1] + summat[r1][c1]
                    if squaresum==(l+1)*(l+1): return (l+1)*(l+1)

        return 0
````

##### 931. Minimum Falling Path Sum

````python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        dp = copy.deepcopy(matrix) #deepcopy(matrix와 dp가 서로 영향안줌) for 2+D lists
        n = len(matrix)
        
        for i in range(1, n):
            for j in range(n):
                dp[i][j] += min(dp[i-1][max(0,j-1)],min(dp[i-1][j],dp[i-1][min(n-1,j+1)]))
        
        return min(dp[-1])
````

##### 174. Dungeon Game

````python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        rlen = len(dungeon)
        clen = len(dungeon[0])
        
        dp = [[10e5 for _ in range(clen+1)] for _ in range(rlen+1)] 
        dp[rlen][clen-1] = dp[rlen-1][clen] = 1
        
        
        for i in range(rlen-1,-1,-1):
            for j in range(clen-1,-1,-1):
                dp[i][j] = max(1, min(dp[i+1][j],dp[i][j+1])-dungeon[i][j])
        return dp[0][0]
````

#### State Machine

##### 714. Best Time to Buy and Sell Stock with Transaction Fee

* dp
  * buy: buy[i] means max profit on day i in stock buy status &#8594; maintain stock/sell stock on day i+1
  * sell: sell[i] means max profit on day i in stock sell status &#8594; do nothing/buy stock on day i+1
* base case
  * On day 0, buy[0] mean we buy stock on day 0 &#8594; buy[0] = -price[0]
  * sell[0] = 0 as we have no stock to sell
* recursive step
  * buy[i] = buy[i-1] (maintain stock) or sell[i-1]-price[i] (buy stock on day i, which no-stock status on day i-1)
* return only sell status because on last days, we should not hold stock &#8594; sell[-1]

````python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        buy = [0 for _ in range(len(prices))]
        sell = [0 for _ in range(len(prices))]
     
        buy[0] = -prices[0]; sell[0]=0
        
        for i in range(1,len(prices)):
            buy[i] = max(buy[i-1],sell[i-1]-prices[i])
            sell[i] = max(sell[i-1],buy[i-1]+prices[i]-fee)
        
        return sell[-1]
````

##### 309. Best Time to Buy and Sell Stock with Cooldown

* buy[i-1], nostock[i-1]-price[i] &#8594; buy[i]
* buy[i-1]+price[i] &#8594; sell[i] : cooldown 1day를 maintain하는 variable
* sell[i-1], nostock[i-1] &#8594; nostock[i]

````python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = [0 for _ in range(len(prices))]
        sell = [0 for _ in range(len(prices))]
        nostock = [0 for _ in range(len(prices))]
        
        buy[0] = -prices[0]; sell[0] = nostock[0] = 0
        
        for i in range(1, len(prices)):
            buy[i] = max(buy[i-1], nostock[i-1]-prices[i])
            sell[i] = buy[i-1] + prices[i]
            nostock[i] = max(sell[i-1], nostock[i-1])
        return max(0,max(sell[-1], nostock[-1]))
````

##### 123. Best Time to Buy and Sell Stock III

<img src="C:\Users\82103\Desktop\Github\Algorithms\img\123.PNG" alt="123" style="zoom:50%;" />

````python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        
        buy1 = -prices[0]
        sell1 = -10e6
        buy2 = -10e6
        sell2 = -10e6
        
        for i in range(1, len(prices)):
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1+prices[i])
            buy2 = max(buy2, sell1-prices[i])
            sell2 = max(sell2, buy2+prices[i])
            
        return sell2
````

##### 188. Best Time to Buy and Sell Stock IV

* Extension from Best Time to Buy and Sell Stock III Q.
  * states increased to k buys and k sells: buy1 &#8594; sell1 &#8594; buy2 &#8594; sell2 ... &#8594; buyk &#8594; sellk

````python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices or not k:
            return 0
        
        sells = [-10e9]*k
        buys = [-10e9]*k
        buys[0] = -prices[0]
        
        for i in range(1, len(prices)):
            buys[0] = max(buys[0], -prices[i])
            sells[0] = max(sells[0], buys[0]+prices[i])
            for j in range(1, k):
                buys[j] = max(buys[j], sells[j-1]-prices[i])
                sells[j] = max(sells[j], buys[j]+prices[i])
        print(buys,sells)
        return max(0, sells[-1])
````

#### Hash + DP

##### 494. Target Sum

````python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        if sum(nums)<S: return 0
        
        #count all zeros and remove them
        zeros = [i for i in range(len(nums)) if nums[i]==0]
        zeroscnt = len(zeros)
        nums = [ele for ele in nums if ele != 0]
        if len(nums)==0: return pow(2,zeroscnt)
        # print(nums, zeroscnt)
        
        dp = [[0 for _ in range(sum(nums)*2+1)] for _ in range(len(nums))]
        mid = sum(nums)
        dp[0][mid+nums[0]] = 1
        dp[0][mid-nums[0]] = 1
        
        for i in range(1,len(nums)):
            for j in range(mid*2+1):
                if j-nums[i]>=0 and j+nums[i]<mid*2+1:
                    dp[i][j] = dp[i-1][j-nums[i]] + dp[i-1][j+nums[i]]
                elif j-nums[i]>=0:
                    dp[i][j] = dp[i-1][j-nums[i]]
                elif j+nums[i]<mid*2+1:
                    dp[i][j] = dp[i-1][j+nums[i]]
                else:
                    continue
                    
        return dp[-1][mid+S]*pow(2,zeroscnt)
````

##### 1027. Longest Arithmetic Subsequence

````python
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        diffdict = defaultdict(lambda:1)
        
        for i,v1 in enumerate(A):
            for j,v2 in enumerate(A[i+1:], start=i+1):
                diffdict[j,v2-v1] = diffdict[i,v2-v1]+1
        return max(diffdict.values())
````

##### 1218. Longest Arithmetic Subsequence of Given Difference

````python
#TLE solution
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        diffdict = defaultdict(lambda:1)
        for i,v1 in enumerate(arr):
            for j,v2 in enumerate(arr[i+1:],i+1):
                if v2-v1==difference:
                    diffdict[j,v2-v1] = diffdict[i,v2-v1]+1
      
        return max(diffdict.values()) if diffdict else 1
````

````python
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = defaultdict(lambda:0)
        for ele in arr:
            dp[ele] = dp[ele-difference]+1
            
        return max(dp.values())
````

#### DFS + DP

##### 576. Out of Boundary Paths

````python
#TLE solution - DFS recurvie
class Solution:
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        self.cnt = 0

        def dfs(i, j, N):
            if N<0:
                return
            
            if i<0 or i>=m or j<0 or j>=n:
                # print(i,j)
                self.cnt += 1
                return

            dfs(i-1,j,N-1)
            dfs(i+1,j,N-1)
            dfs(i,j-1,N-1)
            dfs(i,j+1,N-1)
        
        dfs(i, j, N)
        return self.cnt%1000000007
````

````python
#DP solution: using defaultdict(DFS+memo)
class Solution:
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        dp = defaultdict()
        
        def dfs(i, j, N):    
            if N<0:
                return 0
            
            if i<0 or i>=m or j<0 or j>=n:
                return 1
            
            if (i,j,N) in dp:
                return dp[(i,j,N)]
            
            total = 0
            total += dfs(i-1,j,N-1)
            total += dfs(i+1,j,N-1)
            total += dfs(i,j-1,N-1)
            total += dfs(i,j+1,N-1)
            dp[(i,j,N)] = total
            return total
        print(dp)
        return dfs(i, j, N)%(1000000007)
````

##### 688. Knight Probability in Chessboard

````python
class Solution:
    def knightProbability(self, N: int, K: int, r: int, c: int) -> float:
        memo = defaultdict()
        
        def dfs(K, curr, curc):
            if curr<0 or curr>=N or curc<0 or curc>=N:
                return 0
            if K==0:
                return 1
            if (K,curr,curc) in memo:
                return memo[(K,curr,curc)]
            
            total = dfs(K-1,curr+1,curc+2) + dfs(K-1,curr+1,curc-2) + dfs(K-1,curr-1,curc+2) + dfs(K-1,curr-1,curc-2) + dfs(K-1,curr+2,curc+1) + dfs(K-1,curr+2,curc-1) + dfs(K-1,curr-2,curc+1) + dfs(K-1,curr-2,curc-1)            
            memo[(K,curr,curc)] = total
            return total
        
        return dfs(K,r,c)/(8**K)
````

#### Minimax DP

##### 486. Predict the Winner

* first rec is maximizer turn: max(left **-** rec(left), right **-** rec(right))
* second rec is minimizer turn: it returns negative of return value from first trial, therefore same equation used max(left**-rec(left)**, right-rec(right))
* third rec is maximizer turn: (-)(-) is + 

````python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        # dp = defaultdict()
        @lru_cache(None)
        def rec(left, right):
            # if (left, right) not in dp:
                if left==right:
                    return nums[left]
                else:
                    return  max(nums[left]-rec(left+1,right), nums[right]-rec(left,right-1))
            # return dp[(left,right)]
        
        return rec(0,len(nums)-1)>=0      
````

##### 877. Stone Game

* almost same as question 486

````python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        @lru_cache(None)
        def rec(left,right):
            if left==right:
                return piles[left]
            else:
                return max(piles[left]-rec(left+1,right),piles[right]-rec(left,right-1))
            
        return rec(0,len(piles)-1)>0
````

## Backtracking

##### 131. Palindrome Partitioning

````python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = list()
        
        def isPalindrome(parts):
 i, j= 0, len(parts)-1
            while i<j:
                if parts[i]!=parts[j]:
                    return False
                else:
                    i+=1
                    j-=1
            return True
            
        def backtrack(curList, start):
            if start==len(s):
                result.append(curList[:])
                return
            else:
                for end in range(start, len(s)):
                    if isPalindrome(s[start:end+1]):
                        curList.append(s[start:end+1])
                        backtrack(curList, end+1)
                        curList.pop()
        backtrack([], 0)
        return result
````

* partitioning을 하기위해서 backtracking/dfs 필요 &#8594; s는 유지하되, start를 통한 s의 시작점 조절
* base case: start가 s의 끝점에 도달할시에 완성된 curList를 result에 append
  * 참고: curList[:] 가 아닌 curList[]로 append시 deep copy가 됨으로 모든 element가 같이 바뀜!!!
* append &#8594; backtrack &#8594; pop을 통해 curList을 다른 end값에 사용 가능
* isPalindrome: overlapping subproblems &#8594; DP???

````python
def isPal(self, s):
    return s == s[::-1]
````

Palindrome easier code;;

##### 1239. Maximum Length of a Concatenated String with Unique Characters

* Unique characters?? Set이용하기 &#8594; ''.join(set(string)) == string
  * 참고: python string is immutable: str.append() [X]
  * "".join(set(string))을 쓰니 순서가 random이 된다 => ''.join(OrderedDict.fromkeys(string).keys()) 을 사용 => len(s) == len(set(s)) 쓰는게 더좋음

````python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        self.maxlen = 0
        
        def backtrack(concat):
            if concat==arr:
                return len(arr)
            if len(concat)>self.maxlen:
                self.maxlen=len(concat)

            for s in arr:
                tmp = "".join([concat, s])
                print(tmp)
                if ''.join(OrderedDict.fromkeys(tmp).keys()) == tmp:
                    backtrack(tmp)
                    tmp = concat
                    
        backtrack("")
        return self.maxlen
````

Backtracking has high time complexity => TLE

````python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        maxlen = 0
        unique = ['']
        
        for word in arr:
            for uniq in unique:
                tmp = word + uniq
                if len(tmp) == len(set(tmp)):
                    print(tmp)
                    unique.append(tmp)
                    maxlen = max(maxlen, len(tmp))
        return maxlen
````

##### 46. Permuataions

````python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans = list()
        def rec(start, end):
            if start==end:
                ans.append(nums[:])
            for i in range(start, end+1):
                nums[start], nums[i] = nums[i], nums[start]
                rec(start+1, end)
                nums[start], nums[i] = nums[i], nums[start]
             
        rec(0, len(nums)-1)
        return ans
````



## BFS, DFS

##### 286. Walls and Gates

* Finding optimal path: BFS(queue) was more comfortable for me => deque()
* searching init: gates

````python
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        
        rlen,clen = len(rooms),len(rooms[0])
        
        for r in range(rlen):
            for c in range(clen):
                if rooms[r][c] == 0:
                    queue = deque()
                    visited = set()
                    for i in range(4):
                        queue.append((r+dx[i], c+dy[i], 1))
                    
                    while queue:
                        curr, curc, curd = queue.popleft()
                        if curr<0 or curr>=rlen or curc<0 or curc>=clen or (curr, curc) in visited or rooms[curr][curc] in [0,-1]:
                            continue
                        else:
                            visited.add((curr, curc))
                            rooms[curr][curc] = min(curd, rooms[curr][curc])
                            for i in range(4):
                                queue.append((curr+dx[i], curc+dy[i], curd+1))
````

##### 694. Number of Distinct Islands

* How to distinguish distinct islands' shape? &#8594; every 1의 location을 dfs 시작점 기준으로 relative direction 기록

````python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        rlen, clen = len(grid), len(grid[0])
        islands = []
        seen = set()
 
        
        def isUniqueIsland():
            for island in islands:
                if len(island) != len(curIsland):
                    continue
                for pos1, pos2 in zip(island, curIsland):
                    if pos1 != pos2:
                        break
                else:
                    return False
            return True
        
        def dfs(i, j):
            if i<0 or i>=rlen or j<0 or j>=clen:
                return 
            if (i,j) in seen or grid[i][j]==0:
                return
            print((i,j))
            seen.add((i,j))
            # grid[i][j]==0
            curIsland.append((i-initi, j-initj))
            dfs(i+1,j)
            dfs(i-1,j)
            dfs(i,j+1)
            dfs(i,j-1)

        
        for i in range(rlen):
            for j in range(clen):
                # if grid[i][j] == 1:
                curIsland = list()
                initi, initj = i, j
                dfs(i,j)
                print(curIsland)
                if curIsland and isUniqueIsland():
                    islands.append(curIsland)
        
        print(islands)
        return len(islands)
````

* 참고: unique island인지 loop을 통해 찾는 방법은 inefficient &#8594; use hashing: same islands have equal hash value &#8594; set of islands
  * curIsland가 list()가 아닌 set()
  * Python cannot add set in another set: use frozenset(set) to insert frozenset  in set
* unique island: relative direction이 아닌 path로 판단하는 방법도 있음 (for example, "RLULLR")

## Linked List

##### 445. Add Two Numbers II

````python
#reversing linked list
def reverse(ll):
            prev = None
            cur = ll
            while cur:
                nextt = cur.next
                cur.next = prev
                prev = cur
                cur = nextt
            ll = prev
````

```python
class Solution:
    def reverse(self, ll):
        prev = None
        cur = ll
        while cur:
            nextt = cur.next
            cur.next = prev
            prev = cur
            cur = nextt
        return prev
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        revl1 = self.reverse(l1)
        revl2 = self.reverse(l2)
        
        res = ListNode()
        reshead = res
        isover = False
        while revl1 and revl2:
            tmp = revl1.val + revl2.val + (1 if isover else 0)
            isover=False
            if tmp>=10:
                isover=True
                res.next = ListNode(tmp-10)
            else:
                res.next = ListNode(tmp)
            print(res.val, revl1.val, revl2.val)
            res = res.next; revl1 = revl1.next; revl2 = revl2.next
        while revl1:
            tmp = revl1.val + (1 if isover else 0)
            isover=False
            if tmp>=10:
                isover=True
                res.next = ListNode(tmp-10)
            else:
                res.next = ListNode(tmp)
            print(res.val, revl1.val)
            res = res.next; revl1 = revl1.next
        while revl2:
            tmp = revl2.val + (1 if isover else 0)
            isover=False
            if tmp>=10:
                isover=True
                res.next = ListNode(tmp-10)
            else:
                res.next = ListNode(tmp)
            print(res.val, revl2.val)
            res = res.next; revl2 = revl2.next
        if isover:
            res.next = ListNode(1)
        ans = self.reverse(reshead.next)
        return ans
```

* appraoch: reverse two linked lists &#8594; add them
  * addition approach: with boolean variable *isover*, add 1 if required
  * since two linked lists can have different length, take three while loops (1st: both additions, 2nd: l1 leftover, 3rd: l2 leftover



# String

##### 249. Group Shifted Strings

````python
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        strings.sort(key=lambda s:len(s))
        ans = list()
        group = list([strings[0]])
        length = len(strings[0])
        
        for i in range(1,len(strings)):
            # print(strings[i],group, ans)
            if len(strings[i]) != length:
                group.sort()
                ans.append(group)
                group = list([strings[i]])
                length = len(strings[i])
            else:
                isSameGroup = True
                for j in range(1,length):
                    diff1 = (ord(group[0][j]) - ord(group[0][j-1]))%26
                    diff2 = (ord(strings[i][j]) - ord(strings[i][j-1]))%26
                    if diff1 != diff2:
                        group.sort()
                        ans.append(group)
                        group = list([strings[i]])
                        length = len(strings[i])
                        isSameGroup = False
                        break
                if isSameGroup:
                    group.append(strings[i])
        if group:
            group.sort()
            ans.append(group)
        return ans
````

##### 539. Minimum Time Difference

````python
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        inttime = [int(t[:2])*60+int(t[3:]) for t in timePoints]
        inttime.sort()
        print(inttime)
        mintime = 10e9
        for i in range(1,len(inttime)):
            mintime = min(mintime, inttime[i]-inttime[i-1])
        mintime = min(mintime, (inttime[0]-inttime[-1])%1440)
        print(mintime)
        return mintime
````

# Trees

##### 114. Flatten Binary Tree to Linked List

````python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return root
        
        cur = root
        while cur.left or cur.right:
            if cur.left:
                lefttree = cur.left
                rightnode = cur.right
                cur.right = lefttree
                while lefttree.right:
                    lefttree = lefttree.right
                lefttree.right = rightnode
                cur.left = None
            cur = cur.right
        
        return root
````

##### 337. House Robber III

````python
# TLE solution: recursive approach, by adding LRU cache problem solved
class Solution:
    def rob(self, root: TreeNode) -> int:
        @lru_cache(None)
        def rec(prevrob, root):
            if not root:
                return 0
            if not prevrob:
                return max(rec(1, root.left)+rec(1, root.right)+root.val, rec(0, root.left)+rec(0, root.right))
            else:
                return rec(0, root.left)+rec(0, root.right)
        
        return rec(0, root)
````

##### 199. Binary Tree Right Side View

````python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        reorder = defaultdict()
        
        def traverse(root, depth, position):
            if root:
                if depth not in reorder.keys():
                    reorder[depth] = [(position, root.val)]
                elif (position, root.val) not in reorder[depth]:
                    reorder[depth].append((position, root.val))
                traverse(root.left, depth+1, position+'L')
                traverse(root.right, depth+1, position+'R')
                
 
        traverse(root, 0, "")
        answer = list()
        for k, v in reorder.items():
            v.sort(reverse=True)
            answer.append(v[0][1])
        return answer
````

##### 236. Lowest Common Ancestor of a Binary Tree

````python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        encode = defaultdict()
        
        def traverse(root, path):
            if root:
                if root.val not in encode.keys():
                    encode[root.val] = path
                traverse(root.left, path+"L")
                traverse(root.right, path+"R")
                
        def traverse2(root):
            if root:
                if root.val == ele[0]:
                    return root
                return traverse2(root.left) or traverse2(root.right)
            
        traverse(root, "")
        ecdp = encode[p.val]; ecdq = encode[q.val]
        tog = "";
        for i in range(min(len(ecdp), len(ecdq))):
            if ecdp[i]==ecdq[i]:
                tog += ecdp[i]
            else: break
            
        ele = [k for k,v in encode.items() if v==tog]
        return traverse2(root)
````

##### 103. Binary Tree Zigzag Level Order Traversal

````python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        reorder = defaultdict()
        
        def traverse(root, depth, position):
            if root:
                if depth not in reorder.keys():
                    reorder[depth] = [(position, root.val)]
                elif (position, root.val) not in reorder[depth]:
                    reorder[depth].append((position, root.val))
                traverse(root.left, depth+1, position+'L')
                traverse(root.right, depth+1, position+'R')
                
 
        traverse(root, 0, "")
        print(reorder)
        ans = list()
        
        for k,v in reorder.items():
            if k%2==0:
                v.sort()
            else:
                v.sort(reverse=True)
            filterv = [val for path,val in v]
            ans.append(filterv)
        return ans     
````



# Hash Table

##### 347. Top K Frequent Elements

* Counter: most_common(k): 가장 많이 count된 k개의 (ele, cnt) 반환

````python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        numscnt = Counter(nums)
        return [val for val, cnt in numscnt.most_common(k)]
````

