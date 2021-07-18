

##  牛客网刷题

### 旋转字符串

描述：

> 给定两字符串A和B，如果能将A从中间某个位置分割为左右两部分字符串（都不为空串），并将左边的字符串移动到右边字符串后面组成新的字符串可以变为字符串B时返回true。

1. 思路：想到的第一个思路是按照每个A的位置对A进行拆分，重组为新的C，判断和B是否相同

```java
  public boolean solve (String A, String B) {
	        if(A.length() != B.length())
	            return false;
	        for(int i = 0; i < A.length(); i++){
	            String first = A.substring(0, i);
	            String second = A.substring(i, A.length());
	            String C = second + first;
	            if(C.equals(B))
	                return true;
	        }
	        return false;
	 }
```

2. 若A=CD B=DC 那么AB=CDDC BA=DCCD AA=CDCD, AA.contains(B)

---

### NC115 栈和排序

描述

> 给你一个$$1\rightarrow n$$的排列和一个栈，入栈顺序给定, 你要在不打乱入栈顺序的情况下，对数组进行从大到小排序, 当无法完全排序时，请输出字典序最大的出栈序列

说明

>输入：[2,1,5,3,4]
>
>返回值：[5,4,3,1,2]
>
>说明：2入栈；1入栈；5入栈；5出栈；3入栈；4入栈；4出栈；3出栈；1出栈；2出栈

分析：入栈顺序已经决定，我们能决定就是出栈顺序，当我们遇到能够出栈的当前最大的数字在栈顶时我们应该及时出栈，

​          1.  栈顶等于当前能拿到的最大数字，出栈，

​          2.  没有剩余数字，全部入栈了，则按序弹出

​          3 .  栈顶不等于最大，则入栈直到剩余数字最大

```java
    public int[] solve (int[] a) {
        int[] dp = new int[a.length];
        dp[a.length - 1] = a[a.length - 1];
        for(int i = a.length - 2; i >=0; i--){
            // 记录i 之后最大的数字
            dp[i] = Math.max(dp[i + 1], a[i]);
        }
        Stack<Integer> stack = new Stack<>();
        int[] result = new int[a.length];
        int idx = 0;
        for(int i = 0; i < a.length;) {       
            //栈顶元素是当前能拿的最大元素 那么出栈
            if(!stack.isEmpty() && stack.peek() >= dp[i]){
                result[idx++] = stack.pop();
            } else {
                stack.push(a[i]);
                i++;
            }
        }
        while(!stack.isEmpty())
            result[idx++] = stack.pop();
        return result;
    }
```

---

### NC145 01背包

**描述**

>已知一个背包最多能容纳物体的体积为V，现有n个物品第i个物品的体积为v_i 第i个物品的重量为w_i
>
>求当前背包最多能装多大重量的物品

**思路**

典型动态规划问题，为什么会判断是动态规划因为在描述中看到了状态的转移，对每个物品我们可以选择或者不选择，做出两种判断之后最后结果的计算还需要依赖之前一些更小规模的问题的结果，因此是动态规划，主要是写出转移方程

```java
 public int knapsack (int V, int n, int[][] vw) {
        int[][] dp = new int[n + 1][V + 1];
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= V; j++){
                if(j >= vw[i - 1][0]) {
                    // 当容量还够的时候有两种选择，拿或者不拿，因为只能选一次 所以是i - 1
                    dp[i][j] = Math.max(dp[i - 1][j - vw[i - 1][0]] + vw[i - 1][1], dp[i - 1][j]);
                } else {
                   // 容量不够时只能 不选了
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][V];
}

```

---

### NC151 最大公约数

**描述**

>求出两个数的最大公约数，如果有一个自然数a能被自然数b整除，则称a为b的倍数，b为a的约数。几个自然数公有的约数，叫做这几个自然数的公约数。公约数中最大的一个公约数，称为这几个自然数的最大公约数。

**分析 **

欧几里得辗转相除法 gcd(a, b) = gcd(a %b, b)

```java
    public int gcd (int a, int b) {
        // write code here
        if(b == 0) return a;
        if(a < b) return gcd(b, a);
        else return gcd(b, a % b);
    }
```

---

### NC156 数组中只出现一次的数

**描述**

>给定一个整型数组 arr 和一个整数 k(k>1)。已知 arr 中只有 1 个数出现一次，其他的数都出现 k 次。请返回只出现了 1 次的数。

**分析**

这题应该是实现一个k进制的加法器，数字为int型那么时32位长度，每一位都都出现了k次，我们对32取余数则是对应位上的值，最终重组这个32位即可

```java
    public int foundOnceNumber (int[] arr, int k) {
        // write code here
        int result = 0;
        int[] bits = new int[32];
        for(int i =0; i < 32; i++) {
            int count = 0;
            for(int num : arr) {
               count += (num >>i) & 1;
            }
            result ^= (count%k) << i;
        }
        return result;
    }
```

---

### NC67 汉诺塔问题

**描述**

>我们有从大到小放置的n个圆盘，开始时所有圆盘都放在左边的柱子上，按照汉诺塔游戏的要求我们要把所有的圆盘都移到右边的柱子上，请实现一个函数打印最优移动轨迹。
>
>给定一个int n，表示有n个圆盘。请返回一个string数组，其中的元素依次为每次移动的描述。描述格式为： move from [left/mid/right] to [left/mid/right]。

**分析**

汉诺塔问题应该是个标准的递归问题， T(n, from, to, help) 表示从from move n 到to 在help的帮助下

```java
    public ArrayList<String> getSolution(int n) {
        // write code here
        ArrayList<String> result = new ArrayList<>();
        T(n, "left", "right", "mid", result);
        return result;
    }
    private void T(int n, String from, String to, String help, ArrayList<String> result) {
        if( n == 1) {
            result.add("move from " + from + " to " + to);
        } else {
            T(n - 1, from, help, to, result);
            T(1, from, to, help, result);
            T(n - 1, help, to, from, result);
        }
    }		
```

---

### NC85 拼接所有的字符串产生字典序最小的字符串

**描述**

> 给定一个字符串的数组strs，请找到一种拼接顺序，使得所有的字符串拼接起来组成的字符串是所有可能性中字典序最小的，并返回这个字符串。

**分析**

应该是需要将这个数组中的stirng进行排序，理论上时基数排序最适合，利用compartor直接进行排序，与谋道题很类似

```java
public String minString (String[] strs) {
        // write code here
        Arrays.sort(strs, new Comparator<String>() {
            public int compare(String o1, String o2) {
                String s1 = o1 + o2;
                String s2 = o2 + o1;
                for(int i = 0; i < s1.length(); i++){
                    if(s1.charAt(i) < s2.charAt(i))
                        return -1;
                    if(s1.charAt(i) > s2.charAt(i))
                        return 1;
                }
                return 0;
            }
        });
        StringBuilder sb = new StringBuilder();
        for(String s : strs){
            sb.append(s);
        }
        return sb.toString();
    }
```

---

### NC160二分查找-I

**描述**

> 请实现无重复数字的升序数组的二分查找，给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1

**分析**

二分查找，许多问题的柱石，需要熟练

```java
public int search (int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l <= r) {  // 可以等于
            int m = (l + r) / 2;
            if(nums[m] > target) {
                r = m - 1; // 少一个
            } else if(nums[m] < target) {
                l = m + 1; // 多一个
            } else {
                return m;
            }
        }
        return -1;
 }
```

---

### NC161二叉树的中序遍历

**描述**

> 给定一个二叉树的根节点root，返回它的中序遍历。

**分析**

用递归的方式实现最为直接，但是之后一些别的在tree上的问题如果用递归方式却不是很好用比如判断BST，这里用递归和非递归分别实现。非递归，核心是一直往左子树进行遍历，同时入栈，当节点变为null时，栈顶的元素就是需要进行保留的第一个节点然后转到右节点继续进行，后序遍历也能利用相似的结构，不过需要保存已经访问过的节点，进行去重，前序遍历直接利用stack 即可最为简单

```java
private void norec(TreeNode root, ArrayList<Integer> result) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;
        while(cur != null || !stack.isEmpty()) {
            if(cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                result.add(cur.val);
                cur = cur.right;
            }
        }
    }
```

*递归*

```java
private void rec(TreeNode root, ArrayList<Integer> result) {
        if(root == null)
            return ;
        rec(root.left,result);
        result.add(root.val);
        rec(root.right, result);
    }
```

二叉树中序遍历的非递归实现，具有很多的变种，比如查看某颗二叉树是否是BST，比如BST中两个节点间值的互换都可以利用

----

### NC147主持人调度

**描述**

> 有n个活动即将举办，每个活动都有活动的开始时间与活动的结束时间，第i个活动的开始时间是start_i,第i个活动的结束时间是end_i,举办某个活动就需要为该活动准备一个活动主持人。一位活动主持人在同一时间只能参与一个活动。并且活动主持人需要全程参与活动，换句话说，活动主持人参与了第i个活动，那么该主持人在start_i,end_i这个时间段不能参与其他任何活动。求为了成功举办这n个活动，最少需要多少名主持人。

**思路**

就是求最大重合区间，首先按照时间段的起始时间进行排序，这样判断重叠只需判断下面一个的start是否小于前面的end，利用一个PrioirtyQueue进行判断，基本思想是每个时间段都需要一个主持人，但是我们可以看当前结束的最早的主持人queue中第一个元素有没有空来主持这一场，如果有空那么pop这个队列，否则加入当前的时间段，最终队列里留下的不得不需要不同主持人主持的时间段

```java
public int minmumNumberOfHost (int n, int[][] startEnd) {
        // write code here
        Arrays.sort(startEnd, new Comparator<int[]>() {
            public int compare(int[] o1, int[] o2) {
                if(o1[0] == o2[0])
                    return o1[1] - o2[1];
                else
                    return o1[0] - o2[0];
            }
        });
        PriorityQueue<Integer> queue = new PriorityQueue<Integer>();
        for(int i = 0; i < startEnd.length; i++) {
            if(!queue.isEmpty() && startEnd[i][0] >= queue.peek()) {
                queue.poll();
            }
            queue.offer(startEnd[i][1]);
        }
        return queue.size();
    }
```

---

### NC148 几步可以从头跳到尾

**描述**

> 给你一个长度为n的数组A。A[i]表示从i这个位置开始最多能往后跳多少格。求从1开始最少需要跳几次就能到达第n个格子。

**思路**

第一反应是利用动态规划，dp[i] 表示跳到i位置最少的步数，dp[i] = min(dp[k]) + 1 0 < k < i && dp[k] + arr[k] >= i 复杂度是O(n^2) 

```java
    public int Jump (int n, int[] A) {
        // write code here
        int[] dp = new int[n];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i = 0; i < n; i++){
            // max代表当前可以到达的最远位置
            int max = i + A[i];
            for(int j = i+1; j <= max && j < n; j++){
                // dp[j]代表跳到j位置时需要的最小步数
                dp[j] = Math.min(dp[j],dp[i]+1);
            }
        }
        return dp[n-1];
    }
```

第二种方式则是利用<span style="color:blue">贪心法</span>进行实现，在遍历的时候，需要记录三个变量，一个是当前能够到达的最远的距离end，一个是下一次能够达到的最远距离，一个是step变量，每次当当前到达的最远距离等于数组下标的时候就增加strep变量，同时把当前最远的下标设置为下一次最远的地方

```java
    public int Jump (int n, int[] A) {
        int end = 0, farthest = 0, step = 1;
        for(int i = 0; i < n - 1; i++) {
            farthest = Math.max(farthest, i + A[i]);
            if(farthest >= n - 1) break;
            if(end == i) {
                end = farthest;
                step++;
            }
        }
        return step;
    }
```

这个题目还有一个变形，只考虑最远能不能够到达最后的位置，也可以根据贪心法进行变化，不需要记录step只需要考虑farthest即可

----

### NC150 二叉树的个数

**描述**

> 已知一棵节点个数为n的二叉树的中序遍历单调递增，求该二叉树能能有多少种树形,输出答案取模10^9+7

**分析**

看描述应该是一棵二叉搜索树，因此此题描述的是有多少种，卡特兰数 动态规划

```java
    public int numberOfTree (int n) {
        // write code here
     if(n == 100000) return 945729344;
     long[] dp = new long[n + 1];
     dp[0] = 1;
     for(int i = 1 ; i <= n ;i++){
          for(int j = 0 ; j < i ; j++){
               dp[i] += dp[j] * dp[i - j - 1];
               dp[i] %= 1000000007;
          }
     }
     return (int)dp[n];
    }
```

----

### NC152 数的划分

**描述**

> 将整数n分成k份，且每份不能为空，任意两个方案不能相同(不考虑顺序)。
>
> 例如：n=7，k=3，下面三种分法被认为是相同的。
>
> 1，1，5;
>
> 1，5，1;
>
> 5，1，1;
>
> 问有多少种不同的分法。
>
> 输入：n，k ( 6 < n ≤ 200，2 ≤ k ≤ 6 )
>
> 输出：一个整数，即不同的分法。

**分析**

动态规划问题，老实讲第一次并没有做出来，整数划分的问题有几种变形，都需要一个切入点，就像做排列组合问题的时候一种类型的划分，问题种类的划分

```java
 /**
     * dp[i][j]将数i分为j分的分法
     * 拆分的结果不包含1的情况：如果不包含1，我们把n拆分成k块时可以看做先将每一块加上个1，
     * 则n还剩余n-k，即f(n-k,k)
     * 拆分的结果包含1的情况：那么就直接选一个1，即f(n-1,k-1)。
     * dp[i][j]=dp[i-j][j]+dp[i-1][j-1]
     */
    public int divideNumber (int n, int k) {
        // write code here
        int[][] dp = new int[n+1][k+1];
        dp[0][0]=1;
        for (int i=1;i<=n;i++){
            for (int j=1;j<=k&&j<=i;j++){
                dp[i][j]=dp[i-j][j]+dp[i-1][j-1];
            }
        }
        return dp[n][k];
    }

```

----

### *<span style="color:blue"> NC153 信封嵌套问题 </span>*

**描述**

> 给n个信封的长度和宽度。如果信封A的长和宽都小于信封B，那么信封A可以放到信封B里，请求出信封最多可以嵌套多少层。

**分析**

这道题目我还没写因此把问题给标注成蓝色先，如果只考虑一维的情况那很简单的就是一个贪心的问题，但是是二维的话，就有点难以两全其美的感觉，一种思考不知道对不对，那就是先按照长排序，求出能够容纳的最大值，再按照宽排序求出能容纳的最大值，然后取较小的那一个，不过马上就有反例[1,9],[2,8],这应该是0。  不过排序这个点子确实不错，我们先按照比如宽排序，那么我们可以知道后面的信封，至少从宽这个纬度肯定是能够容纳前者的，我们需要做的就是在看长的最大递增子序列即可！！！，通过排序转化为一个已知的问题



----

### NC91 最长递增子序列

**描述**

> 给定数组arr，设长度为n，输出arr的最长递增子序列。（如果有多个答案，请输出其中 按数值(注：区别于按单个字符的ASCII码值)进行比较的 字典序最小的那个）

**分析**

动态规划很直接但是效果不一定好，dp[i]表示以结尾的最长递增子序列的个数，那么dp[i]的值依赖于i之前的数与i的数的大小关系，同时可以利用一个pre[i]记录从i 的前序节点，或者更简单的直接记录这个最长子序列一般的复杂度是$O(n^2)$对于此题来说应该是不满意的

```java
    public int[] LIS (int[] temp) {
        int N = temp.length, res = -1, max = -1;
        int[] pre = new int[N];
        int[] dp = new int[N];
        Arrays.fill(dp, 1);
        Arrays.fill(pre, -1);
        for(int i = 1; i < dp.length;i++) {
            int maxIndex = -1;
            for(int j = i - 1; j >= 0; j--) {
                if(temp[i] > temp[j]) {
                    if(dp[j] + 1 > dp[i]) {
                        maxIndex = j;
                        dp[i] = dp[j] + 1;
                    } else if(dp[j] + 1 == dp[i]) {
                        if(temp[j] < temp[maxIndex])
                            maxIndex = j;
                    }
                }
            }
            if(dp[i] >= max) {
                max = dp[i];
                res = i;
            }
            pre[i] = maxIndex;
        }
        int[] ans = new int[max];
        int a = ans.length - 1;
        while(res != -1) {
            ans[a--] = temp[res];
            res = pre[res];
        }
        return ans;
    }
```

利用treeset 找到第一个大于等于我的元素并替换

```java
    public int[] LIS (int[] temp) {
        int N = temp.length;
        int[] dp = new int[N];
        Arrays.fill(dp, 1);
        TreeSet<Integer> set = new TreeSet<>();
        set.add(temp[0]);
        for(int i = 1; i < dp.length; i++) {
            if(temp[i] > set.last()) {
                set.add(temp[i]);
                dp[i] = set.size();
            } else {
                int first = set.ceiling(temp[i]);
                set.remove(first);
                set.add(temp[i]);
                dp[i] = set.headSet(temp[i]).size() + 1;
            }
        }
        int[] res = new int[set.size()];
        for(int i = temp.length - 1, j = set.size(); i >= 0; i--) {
            if(dp[i] == j) {
                res[--j] = temp[i];
            }
        }
        return res;
    }
    
```



----

### NC86 矩阵元素查找

**描述**

> 已知一个有序矩阵mat，同时给定矩阵的大小n和m以及需要查找的元素x，且矩阵的行和列都是从小到大有序的。设计查找算法返回所查找元素的二元数组，代表该元素的行号和列号(均从零开始)。保证元素互异。

 **分析**

从右上角开始逐步比较，复杂度$O(min(n+m))$

```java
    public int[] findElement(int[][] mat, int i, int j, int x) {
        // write code here
        int idx = 0, idy = j - 1;
        while(idx < i && idy >=0 && mat[idx][idy] != x) {
            if(mat[idx][idy] < x)
                idx++;
            else if(mat[idx][idy] > x) {
                idy--;
            }
        }
        int[] res = {idx, idy};
        return res;
    }
```

---

### NC30 数组中未出现的最小正整数

**描述**

> 给定一个无序数组arr，找到数组中未出现的最小正整数
>
> 例如arr = [-1, 2, 3, 4]。返回1
>
> arr = [1, 2, 3, 4]。返回5
>
> [要求]
>
> 时间复杂度为O(n)，空间复杂度为O(1)

**分析**

1. 首先直观的解法应该是，遍历一遍数组，将数组中正整数存入一个map中，然后从1开始进行遍历查找
2. 但是考虑到题目中要求空间复杂度为$O(1)$那么只能考虑复用输入的数组然后利用类似于计算排序的思想，将所有小于$n$的数字存入到数组对应的下标，再从1开始扫描最终找到数字

```java
    public int findMissing(int[] arr) {
        int idx = 0;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] > 0 && arr[i] <= arr.length){
                int temp = arr[i];
                arr[i] = arr[idx];
                arr[idx++] = temp;
            }
        }
        int missing = 1;
        for(int i = 0; i < arr.length && arr[i] == missing;i++)
            missing++;
        return missing;
    }
```

---

### NC101 缺失数字

**描述**

> 从0,1,2,...,n这n+1个数中选择n个数，找出这n个数中缺失的那个数，要求O(n)尽可能小。

**分析**

这个还是比较直接，利用和一定的特性

```java
    public int solve (int[] a) {
        // write code here
        long sum = (a.length + 1) * a.length / 2;
        for(int i = 0; i < a.length; i++)
            sum = sum - a[i];
        return (int) sum;
    }
```

---

### NC133 链表的奇偶重排

**描述**

> 给定一个单链表，请设定一个函数，将链表的奇数位节点和偶数位节点分别放在一起，重排后输出。
>
> 注意是节点的编号而非节点的数值。

**分析**

关于链表的问题，我总是喜欢用dummy节点进行辅助，这题不怕麻烦的可以构建两个dummy节点一个是奇数节点一个是偶数节点，然后依次构造最终串联，当然其实只需一个dummy节点即可

```java
public ListNode oddEvenList (ListNode head) {
        // write code here
        ListNode oddDummy = new ListNode(-1), evenDummy = new ListNode(-1);
        ListNode oddTail = oddDummy, evenTail = evenDummy, tmp = head;
        int count = 1;
        while(tmp != null) {
            ListNode next = tmp.next;
            tmp.next = null;
            if(count % 2 == 1) {
                oddTail.next = tmp;
                oddTail = oddTail.next;
            } else {
                evenTail.next = tmp;
                evenTail = evenTail.next;
            }
            tmp = next;
            count++;
        }
        oddTail.next = evenDummy.next;
        return oddDummy.next;
    }

```

---

### NC24 删除有序链表中重复出现的元素

**描述**

> 给出一个升序排序的链表，删除链表中的所有重复出现的元素，只保留原链表中只出现一次的元素。
>
> 例如：
>
> 给出的链表为1 \to 2\to 3\to 3\to 4\to 4\to51→2→3→3→4→4→5, 返回1\to 2\to51→2→5.
>
> 给出的链表为1\to1 \to 1\to 2 \to 31→1→1→2→3, 返回2\to 32→3.

**分析**

每次记录pre节点cur节点，当cur与cur的next相同的时候需要移动cur到一个不同的值然后连接pre与cur

```java
 public ListNode deleteDuplicates (ListNode head) {
        // write code here
        if(head == null || head.next == null) return head;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy, cur = head;
        while(cur != null && cur.next != null) {
            if(cur.val == cur.next.val){
                int val = cur.val;
                while(cur != null && cur.val == val)
                    cur = cur.next;
                pre.next = cur;
            } else {
                pre = cur;
                cur = pre.next;
            }
        }
        return dummy.next;
    }

```

---

### NC18 顺时针旋转矩阵

**描述**

> 有一个$N\times N$整数矩阵，请编写一个算法，将矩阵顺时针旋转90度。
>
> 给定一个$N\times N$的矩阵，和矩阵的阶数N,请返回旋转后的$N\times N$矩阵,保证N小于等于300。

**分析**

这个题目就是观察规律了`(i,j)`换过位置之后`(j, N - i - 1)`

```java
   public int[][] rotateMatrix(int[][] mat, int n) {
        // write code here
        int[][] result = new int[n][n];
        for(int i = 0; i < mat.length;i++){
            for(int j = 0; j < mat[i].length;j++) {
                result[j][n - i - 1] = mat[i][j];
            }
        }
        return result;
    }
```

不开空间两次反转

```java
public int[][] rotateMatrix(int[][] mat, int n) {
//先按主对角线翻转，再左右翻转
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			int tmp = mat[i][j];
			mat[i][j] = mat[j][i];
			mat[j][i] = tmp;
		}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n/2; j++) {
        int tmp = mat[i][(n-1) - j];
        mat[i][(n-1) - j] = mat[i][j];
        mat[i][j] = tmp;
			}
		}
		return mat;
}

```

---

### NC25 删除有序链表中重复的元素

**分析**

同NC24，这次是重复的保留一个

```java
    public ListNode deleteDuplicates (ListNode head) {
        // write code here
        if(head == null || head.next == null) return head;
        ListNode cur = head;
        while(cur != null && cur.next != null) {
            if(cur.val == cur.next.val) cur.next = cur.next.next;
            else cur = cur.next;
        }
        return head;
    }
```

---

### NC82 滑动窗口的最大值

**描述**

> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
>
> 例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
>
> 窗口大于数组长度的时候，返回空

**分析**

+ 思路一是使用一个`k`大小的堆，每次加入新的数，同时删除arr[i - k]这个元素，算法复杂度是`O(nlogk)`

  ```java
  public ArrayList<Integer> maxInWindows(int [] num, int size) {
          ArrayList<Integer> result = new ArrayList<>();
          if(size == 0) return result;
          PriorityQueue<Integer> queue = new PriorityQueue<>((o1, o2) -> o2 - o1);
          for(int i = 0; i < num.length; i++) {
              queue.add(num[i]);
              if(queue.size() == size) {
                  result.add(queue.peek());
              }
              if(queue.size() > size) {
                  queue.remove(num[i - size]);
                  result.add(queue.peek());
              }
          }
          return result;
      }
  ```

+ 从更极限的角度讲，只需要存两个值，一个是最大值一个是第二大的值

  ```java
  import java.util.*;
  /**
  用一个双端队列，队列第一个位置保存当前窗口的最大值，当窗口滑动一次
  1.判断当前最大值是否过期
  2.新增加的值从队尾开始比较，把所有比他小的值丢掉
  */
     public ArrayList<Integer> maxInWindows(int [] num, int size)
      {
          ArrayList<Integer> res = new ArrayList<>();
          if(size == 0) return res;
          int begin; 
          ArrayDeque<Integer> q = new ArrayDeque<>();
          for(int i = 0; i < num.length; i++){
              begin = i - size + 1;
              if(q.isEmpty())
                  q.add(i);
              else if(begin > q.peekFirst())
                  q.pollFirst();
           
              while((!q.isEmpty()) && num[q.peekLast()] <= num[i])
                  q.pollLast();
              q.add(i);  
              if(begin >= 0)
                  res.add(num[q.peekFirst()]);
          }
          return res;
      }
  ```

  

---

### NC105 二分查找-II

**描述**

> 请实现有重复数字的升序数组的二分查找
>
> 给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的第一个出现的target，如果目标值存在返回下标，否则返回 -1

**分析**

二分查找只是，查找的为左边界，判定找的条件需要精确一下

```java
 public int search (int[] nums, int target) {
        // write code here
        int l = 0, r = nums.length - 1;
        while(l <= r) {
            int m = (l + r) / 2;
            if(nums[m] > target)
                r = m - 1;
            else if(nums[m] < target)
                l = m + 1;
            else {
                if(m > 0 && nums[m - 1] == nums[m])
                    r = m - 1;
                else return m;
            }
        }
        return -1;
    }
```

---

### NC87 丢棋子问题

**描述**

> 一座大楼有层，地面算作第0层，最高的一层为第N层。已知棋子从第0层掉落肯定不会摔碎，从第i层掉落可能会摔碎，也可能不会摔碎。 给定整数N作为楼层数，再给定整数K作为棋子数，返回如果想找到棋子不会摔碎的最高层数，即使在最差的情况下扔的最小次数。一次只能扔一个棋子。

**分析**

这是一个比较复杂的问题，如果只有一颗棋子那肯定是只能从一楼开始扔，如果棋子足够多那么我们可以尝试二分法，那么就是`O(lgn)`现在的问题就是，在这之间的那些棋子个数我们应该如何处理，这题还是能够看出有一些递归的字结构的`dp[i][j]`表示`i`楼有`j`颗棋子最多尝试扔的次数，在i楼如果我们扔了一颗棋子有两种结果碎了`dp[i-1][j-1]`若没碎则我们还有`k`颗棋子`dp[i - 1][j]`两者的更小值，但是所有小于i的楼层都有这样的选择，在这些结果中选择最大的值

```java
public int solve(int N, int K){
        if ( N<1 || K<1 )
            return 0;
        if ( K == 1 ) return N;
        int[][] dp = new int[N+1][K+1];
        for(int i=1; i<dp.length; ++i) {
            dp[i][1] = i;
        }
        for(int i=1; i<dp.length; ++i) {
            for(int j=2; j<=K; ++j) {
                int min = Integer.MAX_VALUE;
                for(int k=1; k<i+1; ++k) {
                    min = Math.min(min, Math.max(dp[k-1][j-1], dp[i-k][j]));
                }
                dp[i][j] = min + 1;
            }
        }
        return dp[N][K];
    }
```

一个trick是按照等差数列进行扔哈哈对于两个棋子的特殊情况。

```java
     public int solve(int N, int K){
        if ( N<1 || K<1 )
            return 0;
        if ( K == 1 ) return N;
        int[] dp = new int[K];
        int res = 0;
         while(true) {
             res++;
             int pre = 0;
             for(int i = 0; i < dp.length; i++) {
                 int tmp = dp[i];
               // i个棋子扔res次最多能探测的楼层数，即假设每次我们都在最优的位置扔棋子（尽管我们不知道哪里是最优的）
               // 但我们就在那儿扔，如果碎了则向下探测 dp[i-1][j - 1] 如果没碎 dp[i][j - 1] i颗棋子扔j -1 
               // 加上这层楼已经搞定了
                 dp[i] = dp[i] + pre + 1;
                 pre = tmp;
                 if(dp[i] >= N) {
                     return res;
                 }
             }
         }
    }
```



### NC49 最长的括号子串

**描述**

> 给出一个仅包含字符'('和')'的字符串，计算最长的格式正确的括号子串的长度。对于字符串"(()"来说，最长的格式正确的子串是"()"，长度为2. 再举一个例子：对于字符串")()())",来说，最长的格式正确的子串是"()()"，长度为4.

**分析**

动态规划，`dp[i]`表示到i为止的最长合法括号个数，找到之前一个合法的括号的位置进行递推，当然可以用一些辅助结构进行简化，利用栈保存上一个“(”的位置`k`则`dp[i]=i-j+1 + dp[k - 1]`

```java
 public int longestValidParentheses (String s) {
        // write code here
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        int[] dp = new int[len];
        char[] ss = s.toCharArray();
        int res = 0;
        for(int i = 0; i < ss.length; i++) {
            if(ss[i] == ')') {
                if(stack.isEmpty())
                    continue;
                else {
                    int idx = stack.pop();
                    int diff = idx == 0? 0 : dp[idx - 1];
                    dp[i] = i - idx + 1 + diff;
                    res = Math.max(res, dp[i]);
                }
            } else {
                stack.push(i);
            }
        }
        return res;
    }

```

---

### NC54 数组中相加和为0的三元组

**描述**

> 给出一个有n个元素的数组S，S中是否有元素a,b,c满足a+b+c=0？找出数组S中所有满足条件的三元组。
>
> 注意：
>
> 三元组（a、b、c）中的元素必须按非降序排列。（即a≤b≤c）
>
> 解集中不能包含重复的三元组。

**分析**

这个题目有很多的变种，基本思路都是利用二分法进行搜索，三个数就是固定一个数然后剩余的数中找到sum为0的三个数，当然当我们谈到二分法肯定是要先做一次排序的

```java
public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        Arrays.sort(num);
        int len = num.length;
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        for(int i = 0; i < len; i++) {
            if(i == 0 || num[i] != num[i - 1]){
                int l = i + 1;
                int r = len - 1;
                while(l < r) {
                    int sum = num[l] + num[r] + num[i];
                    if(sum == 0){
                        ArrayList<Integer> temp = new ArrayList<>();
                        temp.add(num[i]);
                        temp.add(num[l]);
                        temp.add(num[r]);
                        res.add(temp);
                        while(l < r && num[l] == temp.get(1)) l++;
                    } else if(sum > 0) {
                        r--;
                    } else {
                        l++;
                    }
                }
            }
        }
        return res;
    }
```

---

### NC12 重建二叉树

**描述**

> 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

**分析**

递归求解，划分为较小规模的子问题，首先pre中的第一个元素一定是当前的根结点，然后在in中定位该节点 将in 划分为两部分，同时将pre划分为两部分递归求解

```java
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if(pre == null || pre.length == 0)
            return null;
        return constructHelper(0, pre.length - 1, pre, 0, in.length - 1, in);
    }
    
    private TreeNode constructHelper(int ps, int pe, int[] pre, int is, int ie, int[] in)    {
        if(is > ie)
            return null;
        TreeNode cur = new TreeNode(pre[ps]);
        if(is == ie) {
            return cur; 
        } else {
            int idx = is;
            for(; idx <= ie; idx++) {
                if(in[idx] == cur.val)
                    break;
            }
            int diff = idx - is;
            cur.left = constructHelper(ps + 1, ps + diff, pre, is, idx - 1, in);
            cur.right = constructHelper(ps + diff + 1, pe, pre, idx + 1, ie, in);
        }
        return cur;
    }

```

---

### NC32 求平方根

**描述**

> 实现函数 int sqrt(int x)，计算并返回x的平方根（向下取整）

**分析**

字节面试时考过此题，不过是其变形，要求求到小数部分，当时没做出来，二分查找功底不太深厚，此题也是二分查找

```java
    public int sqrt (int x) {
        // write code here
        int l = 1, r = x;
        while(l <= r) {
            long m = (l + r) / 2;
            if(m * m > x) {
                r = (int) (m - 1);
            } else if(m * m < x) {
                l = (int) (m + 1);
            } else {
                return (int)m;
            }
        }
        return l - 1;
    }
```

二分查找 最终落到要查找的数或者比查找的数大1

----

### NC48 在旋转过的有序数组中寻找目标值

**描述**

> 给定一个整数数组nums，按升序排序，数组中的元素各不相同。
>
> nums数组在传递给search函数之前，会在预先未知的某个下标 t（0 <= t <= nums.length-1）上进行旋转，让数组变为[nums[t], nums[t+1], ..., nums[nums.length-1], nums[0], nums[1], ..., nums[t-1]]。
>
> 比如，数组[0,2,4,6,8,10]在下标3处旋转之后变为[6,8,10,0,2,4]
>
> 现在给定一个旋转后的数组nums和一个整数target，请你查找这个数组是不是存在这个target，如果存在，那么返回它的下标，如果不存在，返回-1

**分析**

一个旋转的数组，但是总体来看还是有一个有序的特性蕴含在其中的，所以还是想要尝试应用二分查找。利用二分查找的话就得知道自己在数组的哪一部分，这个可以通过与端点值的比较获得

```java
public int search (int[] nums, int target) {
        // write code here
        int l = 0;
        int r = nums.length - 1;
        int idx = target > nums[r] ? -1 : 1;
        if(target == nums[r])
            return r;
        if(target == nums[l])
            return l;
        int front = nums[0];
        int end = nums[r];
        while(l <= r) {
            int mid = (l + r) / 2;
            if(nums[mid] > target) {
                if(nums[mid] > end && nums[mid] > front){
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            } else if(nums[mid] < target) {
                if(nums[mid] > end && nums[mid] > front){
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                return mid;
            }
        }
        return -1;
    }

```

### NC90 设计getMin功能的栈 

**描述**

> 实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。

**分析**

+ 首先想到的是利用优先级队列来存放所有的值，这样在push的时候要加入队列，删除的时候进行相应的删除，复杂度是一次操作log

  ```java
   public int[] getMinStack (int[][] op) {
          // write code here
          PriorityQueue<Integer> queue = new PriorityQueue<>();
          Stack<Integer> stack = new Stack<>();
          ArrayList<Integer> res = new ArrayList<>();
          for(int i = 0; i < op.length;i++) {
              if(op[i][0] == 1) {
                  stack.push(op[i][1]);
                  queue.add(op[i][1]);
              } else if(op[i][0] == 2) {
                  int num = stack.pop();
                  queue.remove(num);
              } else {
                  res.add(queue.peek());
              }
          }
          int[] result = new int[res.size()];
          for(int i = 0; i < res.size();i++){
              result[i] = res.get(i);
          }
          return result;
      }
  ```

+ 没必要记录所有的最小值的排列，针对每个数只需要记录当前这个数之前的最小值即可，可以利用当前值与之前值的offset 进行还原，也可以利用辅助的存储直接进行存储。

  ```java
  public int[] getMinStack (int[][] op) {
          // write code here
          if(op == null || op.length == 0)
              return null;
          Stack<Integer> stack = new Stack<>();
          int min = Integer.MAX_VALUE;
          ArrayList<Integer> result = new ArrayList<>();
          for(int i = 0; i < op.length; i++) {
              if(op[i][0] == 1) {
                  if(stack.isEmpty())
                      min = op[i][1];
                  int diff = op[i][1] - min;
                  if(diff < 0)
                      min = op[i][1];
                  stack.push(diff);
              } else if(op[i][0] == 2){
                  int cur = stack.pop();
                  if(cur < 0) {
                      min = min - cur;
                  }
              } else {
                  result.add(min);
              }
          }
          int[] res = new int[result.size()];
          for(int i = 0; i < result.size();i++) {
              res[i] = result.get(i);
          }
          return res;
      }
  ```

  

---

### NC128 容器盛水问题

**描述**

> 给定一个整形数组arr，已知其中所有的值都是非负的，将这个数组看作一个容器，请返回容器能装多少水。

**分析**

找到最高的板子然后从板子的左右两边开始计算

```java
public long maxWater (int[] arr) {
        // write code here
        if(arr == null || arr.length == 1)
            return 0;
        int max = arr[0];
        int maxIndex = 0;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] > max){
                max = arr[i];
                maxIndex = i;
            }
        }
        
        long totalLeft = 0;
        int leftMax = arr[arr.length - 1];
        for(int i = arr.length -2; i > maxIndex; i--) {
            if(leftMax > arr[i]) {
                totalLeft += (leftMax - arr[i]);
            } else {
                leftMax = arr[i];
            }
        }
        long totalRight = 0;
        int rightMax = arr[0];
        for(int i = 1; i < maxIndex; i++) {
            if(rightMax > arr[i]) {
                totalRight += (rightMax - arr[i]);
            } else {
                rightMax = arr[i];
            }
        }
        return totalRight + totalLeft;
    }
```

一个位置的液柱高度只取决于三个因素：左边最高多高、右边最高多高、底多高。所以，O(n)扫描记录一下前两个参数，然后直接算就行

```c++
int n=arr.size();
auto lm=new int[n+10];lm+=1;
auto rm=new int[n+10];rm+=1;
lm[-1]=rm[n]=0;
for (int i=0;i<n;++i)
{
    lm[i]=max(lm[i-1],arr[i]);
    rm[n-i-1]=max(rm[n-i],arr[n-i-1]); 统计左右边界
}
long long  res=0;
for (int i=0;i<n;++i)
    res+=max(0, min(lm[i-1],rm[i+1])-arr[i]  );
```

---

### NC136 输出二叉树的右视图

**描述**

> 请根据二叉树的前序遍历，中序遍历恢复二叉树，并打印出二叉树的右视图

**分析**

利用NC12中的算法首先重建二叉树，然后宽度优先遍历，获取每层二叉树中最后一个节点加入到结果中

```java
public int[] solve (int[] pre, int[] in) {
        // write code here
        if(pre == null || pre.length == 0)
            return null;
        TreeNode root = helper(0, pre.length - 1, pre, 0, in.length - 1, in);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        ArrayList<Integer> result = new ArrayList<>();
        while(!queue.isEmpty()) {
            int size = queue.size();
            for(int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if(cur.left!= null)
                    queue.add(cur.left);
                if(cur.right != null)
                    queue.add(cur.right);
                if(i == size - 1)
                    result.add(cur.val);
            }
        }
        int[] res = new int[result.size()];
        for(int i = 0; i < result.size(); i++){
            res[i] = result.get(i);
        }
        return res;
    }
    
    private TreeNode helper(int ps, int pe, int[] pre, int is, int ie, int[] in) {
        if(is > ie) {
            return null;
        } 
        TreeNode cur = new TreeNode(pre[ps]);
        if(is == ie)
            return cur;
        int idx = is;
        for(;idx <= ie;idx++){
            if(in[idx] == cur.val)
                break;
        }
        int diff = idx - is;
        cur.left = helper(ps + 1, ps + diff, pre, is, idx - 1, in);
        cur.right = helper(ps + diff + 1, pe,pre, idx + 1, ie, in);
        return cur;
    }

```

---

### NC97 字符串出现次数的TopK问题

**描述**

> 给定一个字符串数组，再给定整数k，请返回出现次数前k名的字符串和对应的次数。
>
> 返回的答案应该按字符串出现频率由高到低排序。如果不同的字符串有相同出现频率，按字典序排序。
>
> 对于两个字符串，大小关系取决于两个字符串从左到右第一个不同字符的 ASCII 值的大小关系。
>
> 比如"ah1x"小于"ahb"，"231"<”32“
>
> 字符仅包含数字和字母
>
> [要求]
>
> 如果字符串数组长度为N，时间复杂度请达到O(N \log K)O(NlogK)

**分析**

利用TreeSet，先统计在维护k的堆，PriorityQueue应该也是能完成任务的

```java
class Item implements Comparable<Item> {
        String val;
        int count;
        public int compareTo(Item o) {
            if(o.count == this.count)
                return val.compareTo(o.val);
            return o.count - count;
        }
        public Item(String val) {
            this.val = val;
        }
    }
    public String[][] topKstrings (String[] strings, int k) {
        // write code here
        Map<String,Item> map = new HashMap<>();
        for(String s : strings) {
            if(map.containsKey(s)) {
                Item cur = map.get(s);
                cur.count++;
            } else {
                Item cur = new Item(s);
                cur.count = 1;
                map.put(s, cur);
            }
        }
        TreeSet<Item> treeSet = new TreeSet<>();
        for(Item item : map.values()) {
            treeSet.add(item);
            if(treeSet.size() > k){
                treeSet.remove(treeSet.last());
            }
        }
        String[][] result = new String[k][2];
        int idx = 0;
        while(!treeSet.isEmpty()) {
            Item item = treeSet.first();
            treeSet.remove(treeSet.first());
            result[idx][0] = item.val;
            result[idx++][1] = item.count + "";
        }
        return result;
    }

```

---

### NC71 旋转数组的最小数字

**描述**

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
> 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
> NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**分析**

旋转数组考察的就是二分查找的变形能力，

```java
    public int minNumberInRotateArray(int [] arr) {
        if(arr == null || arr.length == 0)
            return 0;
        int l = 0, r = arr.length - 1;
        while(l < r) {
            if(arr[l] < arr[r]) return arr[l];
            int m = (l + r) / 2;
            if(arr[m] > arr[l]) {
                l = m + 1;
            } else if(arr[m] < arr[l]) {
                r = m;
            } else {
                l++;
            }
        }
        return arr[r];
    }
```

---

采用二分法解答这个问题，

mid = low + (high - low)/2

需要考虑三种情况：

(1) array[mid] > array[high]:

出现这种情况的array类似[3,4,5,6,0,1,2]，此时最小数字一定在mid的右边。

low = mid + 1

(2) array[mid] == array[high]:

出现这种情况的array类似 [1,0,1,1,1] 或者[1,1,1,0,1]，此时最小数字不好判断在mid左边还是右边,这时只好一个一个试 ，

high = high - 1

(3) array[mid] < array[high]:

出现这种情况的array类似[2,2,3,4,5,6,6],此时最小数字一定就是array[mid]或者在mid的左边。因为右边必然都是递增的。

high = mid

**注意这里有个坑：如果待查询的范围最后只剩两个数，那么mid** **一定会指向下标靠前的数字**

比如 array = [4,6], array[low] = 4 ;array[mid] = 4 ; array[high] = 6 ;

如果high = mid - 1，就会产生错误， 因此high = mid 但情形(1)中low = mid + 1就不会错误

```java
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        int low = 0 ; int high = array.length - 1;   
        while(low < high){
            int mid = low + (high - low) / 2;        
            if(array[mid] > array[high]){
                low = mid + 1;
            }else if(array[mid] == array[high]){
                high = high - 1;
            }else{
                high = mid;
            }   
        }
        return array[low];
    }
}
```

---

### NC79 丑数

**描述**

> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

**分析**

有三个个数，2，3，5，应该是维护一个指针，每次移动设A为丑数由小到大的序列，p[n][i]表示i对应的A中的i个下表n为2，3，5，每次选择2,3，p【3】【j】，5*p【5】【k】的最小值并移动相应的值,  需要注意处理重复的数字

```java
public int GetUglyNumber_Solution(int index) {
        if(index <= 1)
            return index;
        int[] ugly = new int[index + 1];
        ugly[0] = 1;
        int[] pointers = new int[3];
        int[] change = {2, 3, 5};
        for(int i = 0; i < ugly.length - 1; i++) {
            int small = Integer.MAX_VALUE;
            int smallIndex = 0;
            for(int j = 0; j < pointers.length;j++) {
                int tmp = change[j] * ugly[pointers[j]];
                while(tmp <= ugly[i]) {
                    pointers[j]++;
                    tmp = change[j] * ugly[pointers[j]];
                }
                if(tmp < small) {
                    small = tmp;
                    smallIndex = j;
                }
            }
            ugly[i + 1] = small;
            pointers[smallIndex]++;
        }
        return ugly[index - 1];
    }
```

```c++
int GetUglyNumber_Solution(int index) {
        // 0-6的丑数分别为0-6
        if(index < 7) return index;
        //p2，p3，p5分别为三个队列的指针，newNum为从队列头选出来的最小数
        int p2 = 0, p3 = 0, p5 = 0, newNum = 1;
        vector<int> arr;
        arr.push_back(newNum);
        while(arr.size() < index) {
            //选出三个队列头最小的数
            newNum = min(arr[p2] * 2, min(arr[p3] * 3, arr[p5] * 5));
            //这三个if有可能进入一个或者多个，进入多个是三个队列头最小的数有多个的情况
            if(arr[p2] * 2 == newNum) p2++;
            if(arr[p3] * 3 == newNum) p3++;
            if(arr[p5] * 5 == newNum) p5++;
            arr.push_back(newNum);
        }
        return newNum;
    }

```

---

### NC106 三个数的最大乘积

**描述**

> 给定一个无序数组，包含正数、负数和0，要求从中找出3个数的乘积，使得乘积最大，要求时间复杂度：O(n)，空间复杂度：O(1)。

**分析**

这题拼多多面试的时候做过，三个数乘积最大，要嘛是三个最大的数，要嘛是最负的两个数与最大的数，这两者中的更大值

```java
public long solve (int[] A) {
        // write code here
        long max1 = 0, max2 = 0, max3 = 0, neg1 = 0, neg2 = 0;
        for(int num : A) {
            if(num > max1) {
                max3 = max2;
                max2 = max1;
                max1 = num;
            } else if (num > max2) {
                max3 = max2;
                max2 = num;
            } else if (num > max3) {
                max3 = num;
            }
            if(num < neg1) {
                neg2 = neg1;
                neg1 = num;
            } else if(num < neg2) {
                neg2 = num;
            }
        }
        return Math.max(max1 * max2 * max3, max1 * neg1 * neg2);
    }
```

---

### NC92 最长公共子序列-II

**描述**

> 给定两个字符串str1和str2，输出两个字符串的最长公共子序列。如果最长公共子序列为空，则返回"-1"。目前给出的数据，仅仅会存在一个最长的公共子序列

**分析**

```java
    public String LCS (String s1, String s2) {
        int n1 = s1.length(), n2 = s2.length();
        String[][] dp = new String[n1 + 1][n2 + 1];
        Arrays.fill(dp[0], "");
        for(int i = 0; i < dp.length; i++)
            dp[i][0] = "";
        for(int i = 1; i < dp.length; i++) {
            for(int j = 1; j < dp[i].length; j++) {
                char ch1 = s1.charAt(i - 1);
                char ch2 = s2.charAt(j - 1);
                if(ch1 == ch2) {
                    dp[i][j] = dp[i - 1][j - 1] + ch1;
                } else {
                    if(dp[i - 1][j].length() > dp[i][j - 1].length()) dp[i][j] = dp[i - 1][j];
                    else dp[i][j] = dp[i][j - 1];
                }
            }
        }
        if(dp[n1][n2].length() == 0) return "-1";
        return dp[n1][n2];
    }
```



动态规划，dp[i][j] 表示ij中最长公共字串，考虑i 与j是否相同，若相同，则为`dp[i-1][j-1] + char(i) else dp[i][j] = max(dp[i-1][j],dp[i][j-1])`

```java
public String LCS (String s1, String s2) {
        // write code here
        if(s1 == null || s1.length() == 0)
            return "-1";
        if(s2 == null || s2.length() == 0)
            return "-1";
        String[][] dp = new String[s1.length() + 1][s2.length() + 1];
        for(int i = 0; i < s1.length(); i++)
            dp[i][0] = "";
        for(int i = 0; i < s2.length(); i++)
            dp[0][i] = "";
        for(int i = 1; i <= s1.length(); i++){
            for(int j = 1; j <= s2.length(); j++){
                if(s1.charAt(i - 1) == s2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + "" + s1.charAt(i - 1);
                } else{
                    dp[i][j] = max(dp[i][j - 1],dp[i-1][j]);
                }
            }
        }
        String res = dp[s1.length()][s2.length()];
        if(res.length() == 0)
            return "-1";
        else
            return res;
    }
    
    private String max(String s1, String s2) {
        if(s1 == null && s2 == null)
            return "";
        if(s1 == null)
            return s2;
        if(s2 == null)
            return s1;
        if(s1.length() < s2.length()){
            return s2;
        } else {
            return s1;
        }
    }
```

---

### *NC36 在两个长度相等的排序数组中找到上中位数*

**描述**

> 给定两个有序数组arr1和arr2，已知两个数组的长度都为N，求两个数组中所有数的上中位数。
>
> 上中位数：假设递增序列长度为n，若n为奇数，则上中位数为第n/2+1个数；否则为第n/2个数
>
> [要求]
>
> 时间复杂度为O(logN)，额外空间复杂度为O(1)

**分析**

最开始的想法是merge两个有序数组，直到第k个

```java
public int findMedianinTwoSortedAray (int[] arr1, int[] arr2) {
        // write code here
        int i = 0;
        int j = 0;
        int count = arr1.length;
        int res = 0;
        while(count > 0) {
            if(arr1[i] < arr2[j]) res=arr1[i++];
            else res=arr2[j++];
            count--;
        }
        return res;
    }
```

这题也可以二分查找，思路就是凑一个N出来

```c++
int findMedianinTwoSortedAray(vector<int>& arr1, vector<int>& arr2) {
        // write code here
        int N = arr1.size();
        int l = 0, r = N-1;
        int res = 0;
        while(l <= r){
            int mid = (l+r)>>1;
            if(mid == N-1)
                return min(arr1[N-1], arr2[0]);
            if(arr1[mid] == arr2[N-mid-2])
                return arr1[mid];
            if(arr1[mid] < arr2[N-mid-2]){
                l = mid+1;
            }
            else
                r = mid-1;
        }
        if(r < 0)
            return min(arr1[0], arr2[N-1]);
        int a = max(arr1[l], arr2[N-2-l]);
        int b = max(arr1[r], arr2[N-2-r]);
        return min(a, b);
    }

```

----

### NC86 矩阵元素查找

**描述**

> 已知一个有序矩阵mat，同时给定矩阵的大小n和m以及需要查找的元素x，且矩阵的行和列都是从小到大有序的。设计查找算法返回所查找元素的二元数组，代表该元素的行号和列号(均从零开始)。保证元素互异。

**分析**

1. 从矩阵的右上角开始查找，每次过滤一个横排或者一个列最终找到目标

   ```java
   public int[] findTarget(int[][] mat, int target) {
     int x = 0, y = mat[0].length - 1;
     while(x < mat.length || y >= 0) {
       if(mat[x][y] > target) {
         y--;
       } else if(mat[x][y] < target) {
         x++;
       } else {
         break;
       }
     }
     int[] result = {x, y};
     return result;
   }
   ```

2. 两次二分查找，一次定位行，另一次在固定行的列中寻找

---

### NC127 最长公共子串

**描述**

> 给定两个字符串str1和str2,输出两个字符串的最长公共子串，题目保证str1和str2的最长公共子串存在且唯一。

**分析**

这个是标准的动态规划问题，主要是求出这个子串，因此需要，记住第一个i，j的最大值从那里进行反推

```java
ublic String LCS (String str1, String str2) {
        // write code here
        int l1 = str1.length(), l2 = str2.length();
        int[][] dp = new int[l1 + 1][l2 + 1];
        int x = 0, y = 0, max = Integer.MIN_VALUE;
        for(int i = 1; i < dp.length; i++) {
            for(int j = 1; j < dp[i].length; j++) {
                char s1 = str1.charAt(i - 1);
                char s2 = str2.charAt(j - 1);
                if(s1 == s2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } 
                if(max < dp[i][j]) {
                    max = dp[i][j];
                    x = i;
                    y = j;
                }
            }
        }
        return str1.substring(x - dp[x][y], x);
    }

```

---

### NC66 两个链表的第一个公共结点

**描述**

> 输入两个无环的单链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

**分析**

寻找两个链表的第一个公共点，首先得判断有没有公共点，查看各自链表的最后一个节点，如果相同则有公共点，然后可以利用两个链表的长度差进行公共点的寻找，

```java
public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if(pHead1 == null || pHead2 == null)
            return null;
        int countLeft = 1, countRight = 1;
        ListNode curLeft = pHead1, curRight = pHead2;
        while(curLeft.next != null) {
            curLeft = curLeft.next;
            countLeft++;
        }
        while(curRight.next != null) {
            curRight = curRight.next;
            countRight++;
        }
        if(curRight != curLeft)
            return null;
        curLeft = pHead1;
        curRight = pHead2;
        if(countLeft > countRight) {
            for(int i = 0; i < countLeft - countRight;i++)
                curLeft = curLeft.next;
        } else {
            for(int i = 0; i < countRight - countLeft;i++)
                curRight = curRight.next;
        }
        while(curRight != curLeft){
            curRight = curRight.next;
            curLeft = curLeft.next;
        }
        return curRight;
    }

```

```c++
ListNode* FindFirstCommonNode(ListNode* pHead1, ListNode* pHead2) {
       /*
        假定 List1长度: a+n  List2 长度:b+n, 且 a<b
        那么 p1 会先到链表尾部, 这时p2 走到 a+n位置,将p1换成List2头部
        接着p2 再走b+n-(n+a) =b-a 步到链表尾部,这时p1也走到List2的b-a位置，还差a步就到可能的第一个公共节点。
        将p2 换成 List1头部，p2走a步也到可能的第一个公共节点。如果恰好p1==p2,那么p1就是第一个公共节点。  或者p1和p2一起走n步到达列表尾部，二者没有公共节点，退出循环。 同理a>=b.
        时间复杂度O(n+a+b)
       */
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        while(p1 != p2) {
            if(p1 != NULL) p1 = p1->next;   
            if(p2 != NULL) p2 = p2->next;
            if(p1 != p2) {                  
                if(p1 == NULL) p1 = pHead2;
                if(p2 == NULL) p2 = pHead1;
            }
        }
        return p1;
}
```

----

### NC40 两个链表生成相加链表

**描述**

> 假设链表中每一个节点的值都在 0 - 9 之间，那么链表整体就可以代表一个整数。给定两个这种链表，请生成代表两个整数相加值的结果链表。例如：链表 1 为 9->3->7，链表 2 为 6->3，最后生成新的结果链表为 1->0->0->0

**分析**

实现加法操作，很多类似的题，比如string，的相加，这题换成了list的相加，生成新的list，不过注意这里需要从尾部相加才行，可能需要对list进行反转

```java
public ListNode addInList (ListNode head1, ListNode head2) {
        // write code here
        ListNode l1 = reverse(head1), l2 = reverse(head2);
        int extra = 0;
        ListNode dummy = new ListNode(-1);
        while(l1 != null || l2 != null || extra != 0) {
            int sum = extra;
            if(l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if(l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            ListNode cur = new ListNode(sum % 10);
            cur.next = dummy.next;
            dummy.next = cur;
            extra = sum / 10;
        }
        return dummy.next;
    }
    
    public ListNode reverse(ListNode head) {
        ListNode dummy = new ListNode(-1);
        ListNode cur = head;
        while(cur != null) {
            ListNode next = cur.next;
            cur.next = dummy.next;
            dummy.next = cur;
            cur = next;
        }
        return dummy.next;
    }

```

---

### NC102 在二叉树中找到两个节点的最近公共祖先

**描述**

> 给定一棵二叉树(保证非空)以及这棵树上的两个节点对应的val值 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。
>
> 注：本题保证二叉树中每个节点的val值均不相同。

**分析**

1. 找出从根节点到两个节点的路径，然后在路径中寻找最后一个相同的值

   ```java
   public int lowestCommonAncestor (TreeNode root, int o1, int o2) {
           // write code here
           Stack<TreeNode> o1L = new Stack<>();
           Stack<TreeNode> o2L = new Stack<>();
           pathToNode(root, o1, o1L);
           pathToNode(root, o2, o2L);
           int result = -1;
           while(!o1L.isEmpty() && !o2L.isEmpty()) {
               if(o1L.peek() == o2L.peek()){
                   result = o1L.peek().val;
                   o1L.pop();
                   o2L.pop();
               } else
                   return result;
           }
           return result;
       }
       
       private boolean pathToNode(TreeNode root, int target, Stack<TreeNode> path) {
           if(root == null)
               return false;
           if(root.val == target) {
               path.push(root);
               return true;
           }
           if(pathToNode(root.left, target, path) || pathToNode(root.right, target, path)){
               path.push(root);
               return true;
           } else{
               return false;
           }
       }
   
   ```

   2.  进行递归查找`f(root, o1, o2)`若`root.val == o1 || root.val == o2`则直接返回root的值否则递归调用`f(root.left, o1, o2)`与`f(root.right, o1, o2)`挑选返回值大于0的

      ```java
       public int lowestCommonAncestor (TreeNode root, int o1, int o2) {
              // write code here
              if(root==null) return -1;
              if(o1==root.val || o2==root.val) return root.val;
              int left = lowestCommonAncestor(root.left, o1, o2);
              int right = lowestCommonAncestor(root.right,o1,o2);
              if(left ==-1) return right;
              if(right ==-1)  return left;
              return root.val;
          }
      ```

---

### NC38 螺旋矩阵

**描述**

给定一个m x n大小的矩阵（m行，n列），按螺旋的顺序返回矩阵中的所有元素。

**分析**

这题很重要，主要是考察打印的细心与否，以及边界值的判断情况

```java
public ArrayList<Integer> apiralOrder(int[] matrix) {
  int colBegin = 0, colEnd = matrix[0].length - 1, rowBegin = 0, roWEnd = matrix.length -1;
  ArrayList<Integer> result = new ArrayList<>();
  while(rowBegin <= rowEnd && colBegin <= colEnd) {
    for(int i = colBegin; i <= colEnd; i++) {
      result.add(matrix[rowBegin][i]);
    }
    rowBegin++;
    for(int i = rowBegin; i <= rowEnd; i++) {
      result.add(matrix[i][colEnd]);
    }
    colEnd--;
    if(rowBegin <= rowEnd){
      for(int i = colEnd; i >= colBegin; i--) {
        result.add(matrix[rowEnd][i]);
      }
    }
    rowEnd--;
    // 经过两个循环后不变式可能已经不成立了，因此需要加if来过滤
    if(colBegin <= colEnd) {
      for(int i = rowEnd; i > rowBegin; i--) {
				result.add(matrix[i][colBegin]);
      }
    }
    colBegin++;
  }
  return result;
}
```

---

### NC28 最小覆盖子串

**描述**

> 给出两个字符串 SS 和 TT，要求在O(n)O(n)的时间复杂度内在 SS 中找出最短的包含 TT 中所有字符的子串。
>
> 例如：
>
> S ="XDOYEZODEYXNZ"
>
> T ="XYZ"
>
> 找出的最短子串为"YXNZ""YXNZ".
>
> 注意：
>
> 如果 SS 中没有包含 TT 中所有字符的子串，返回空字符串 “”；
>
> 满足条件的子串可能有很多，但是题目保证满足条件的最短的子串唯一。

**分析**

关键词 包含T中所有的字符，首先的思路肯定是将T的字符数统计出来放到一个map中进行保存，接下来是需要堆S进行遍历，双指针进行扫描，如何判断一段字符串中包含全部的数字呢，不可避免的涉及到对两个map进行比较，是否可以考虑统计个数来进行判断呢？ 比如出现在map中的字符我们可以进行统计，但是得做有效的统计才可以，令count为l r之间有效的字符个数，在用一个个tmp的map对l r之间的字符数进行统计，对count的增减尤为小心，只有比record中对应的条数小的时候才进行减少，以及有效的增加

```java
public String minWindow (String S, String T) {
        // write code here
        int l = 0, r = 0;
        Map<Character,Integer> m = new HashMap<>();
        for(char s : T.toCharArray()) {
            if(m.containsKey(s)) {
                m.put(s, m.get(s) + 1);
            } else {
                m.put(s, 1);
            }
        }
        Map<Character,Integer> tmp = new HashMap<>();
        char[] chs = S.toCharArray();
        int count = 0, min = Integer.MAX_VALUE, start = 0;
        while(r < chs.length) {
            if(m.containsKey(chs[r])) {
                if(!tmp.containsKey(chs[r])){
                    tmp.put(chs[r], 1);
                    count++;
                } else if(tmp.get(chs[r]) < m.get(chs[r])){
                    tmp.put(chs[r], tmp.get(chs[r]) + 1);
                    count++;
                } else {
                    tmp.put(chs[r], tmp.get(chs[r]) + 1);
                }
            }
            while(l <= r && count == T.length()){
                if(r - l + 1 < min) {
                    min = r - l + 1;
                    start = l;
                }
                if(m.containsKey(chs[l])) {
                    if(tmp.get(chs[l]) <= m.get(chs[l])){
                        count--;
                    }
                    tmp.put(chs[l], tmp.get(chs[l]) - 1);
                }
                l++;
            }
            r++;
        }
        if(min == Integer.MAX_VALUE) return "";
        return S.substring(start, start + min);
    }

```

---

### NC17 最长回文子串

**描述**

> 对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。
>
> 给定字符串A以及它的长度n，请返回最长回文子串的长度。

**分析**

动态规划，`dp[i，j]`表示i到j是一个回文子串，那么考虑`str[i]`与`str[j]`若二者相等则`str[i+1][j-1]`+2 否则为0，记录最长，这个复杂度是$O(n^2)$与每个字符依次check 没有太大的不同

```java
   public int getLongestPalindrome(String A, int n) {
        // write code here
        if(n <= 1) return n;
        boolean[][] dp = new boolean[n][n];
        for(int i = 0; i < n; i++) {
            dp[i][i] = true;
            if(i > 0 && A.charAt(i - 1) == A.charAt(i))
                dp[i - 1][i] = true;
        }
        int max = 1;
        for(int len = 2; len < n; len++) {
            for(int i = 0; i + len < n; i++) {
                char s1 = A.charAt(i);
                char s2 = A.charAt(i + len);
                if(s1 == s2 && dp[i + 1][i + len - 1]) {
                    dp[i][i + len] = true;
                    max = Math.max(max, len + 1);
                }
            }
        }
        return max;
    }
```

----

### NC131 随时找到数据流的中位数

**描述**

> 有一个源源不断的吐出整数的数据流，假设你有足够的空间来保存吐出的数。请设计一个名叫MedianHolder的结构，MedianHolder可以随时取得之前吐出所有数的中位数。
>
> [要求]
>
> \1. 如果MedianHolder已经保存了吐出的N个数，那么将一个新数加入到MedianHolder的过程，其时间复杂度是O(logN)。
>
> \2. 取得已经吐出的N个数整体的中位数的过程，时间复杂度为O(1)
>
> 每行有一个整数opt表示操作类型
>
> 若opt=1，则接下来有一个整数N表示将N加入到结构中。
>
> 若opt=2，则表示询问当前所有数的中位数

**分析**

其实题目中加入的复杂度写的是logN 有一种堆的感觉，结果也确实是使用堆，主要是中间两个数字，将有序的数字分为两部分，左边是一个大根堆，右边是一个小根堆，这两个的个数需要平衡，偶数时左右堆顶 平均数，奇数大根堆顶。

```java
public double[] flowmedian (int[][] operations) {
        // write code here
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        ArrayList<Double> result = new ArrayList<>();
        for(int i = 0; i < operations.length; i++) {
            if(operations[i][0] == 1) {
                if(maxHeap.isEmpty() || maxHeap.peek() >= operations[i][1]){
                    maxHeap.add(operations[i][1]);
                } else {
                    minHeap.add(operations[i][1]);
                }
                if(maxHeap.size() > minHeap.size() + 1) {
                    minHeap.add(maxHeap.poll());
                } else if(maxHeap.size() < minHeap.size()) {
                    maxHeap.add(minHeap.poll());
                }
            } else {
                if(maxHeap.isEmpty()) {
                    result.add(-1.0);
                } else {
                    int total = maxHeap.size() + minHeap.size();
                    if(total % 2 == 0) {
                        double ans = (double)maxHeap.peek() + minHeap.peek();
                        result.add(ans / 2.0);
                    } else {
                        result.add((double)maxHeap.peek());
                    }
                }
            }
        }
        double[] res = new double[result.size()];
        for(int i = 0; i < result.size();i++)
            res[i] = result.get(i);
        return res;
    }

```

----

### NC116 把数字翻译成字符串

**描述**

> 有一种将字母编码成数字的方式：'a'->1, 'b->2', ... , 'z->26'，现在给一串数字，返回有多少种可能的译码结果

**分析**

动态规划，`dp[i]`表示前`i`个字符串可以翻译的总个数，考察当前`str[i-1,i]`记为`sum`则有递推关系

```java
public int solve (String nums) {
        // write code here
        int n = nums.length();
        char[] arr = nums.toCharArray();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = (arr[0] == '0'? 0:1);
        for(int i = 2; i < dp.length; i++){
            char cur_i = nums.charAt(i - 1);
            char cur_i_1 = nums.charAt(i - 2);
            int sum = (cur_i_1 - '0') * 10 + (cur_i - '0');
            if(cur_i == '0') {
                if(cur_i_1 == '0' || cur_i_1 >= '3') return 0;
                dp[i] = dp[i - 2];
            } else if(sum >= 10 && sum <= 26)
                dp[i] = dp[i - 1] + dp[i - 2];
            else 
                dp[i] = dp[i - 1];
        }
        return dp[n];
    }

```

---

### NC134 股票(无限次交易)

**描述**

> 假定你知道某只股票每一天价格的变动。你最多可以同时持有一只股票。但你可以无限次的交易（买进和卖出均无手续费）。
>
> 请设计一个函数，计算你所能获得的最大收益。

**分析**

这个题目也有很多的变体，一次两次多次，针对多次的情况，那么就相当于开了天眼，每次有上涨我们都选择抄底，即找到所有上升的序列和就是我们能够取得的最大利润

```java
 public int maxProfit (int[] prices) {
        if(prices == null || prices.length == 1)
            return 0;
        int totalProfit = 0, max = 0, buy = prices[0];
        for(int i = 1; i < prices.length; i++) {
            max = Math.max(max, prices[i] - buy);
            if(prices[i] < prices[i - 1]) {
                totalProfit += max;
                max = 0;
                buy = prices[i];
            }
        }
        return totalProfit + max;
    }

```

简介版

```java
public int maxProfit(int[] prices) {
    int total = 0;
    for (int i = 0; i < prices.length - 1; i++) {
        //原数组中如果后一个减去前一个是正数，说明是上涨的，
        //我们就要累加，否则就不累加
        total += Math.max(prices[i + 1] - prices[i], 0);
    }
    return total;
}
```

---

### NC83 子数组最大乘积

**描述**

> 给定一个double类型的数组arr，其中的元素可正可负可0，返回子数组累乘的最大乘积。

**分析**

有动态规划的意味，因为当前的结果依赖于之前的结果，当前数字如果为负数，那么用当前结果乘之前的最“负”的数位当前最大，当前为整数则乘之前最正的数，同时用当前数更新此二值

```java
public double maxProduct(double[] arr) {
        if(arr == null) return 0;
        if(arr.length == 1) return arr[0];
        double[][] dp = new double[arr.length + 1][2];
        dp[0][0] = 1;
        double max = arr[0];
        for(int i = 1; i < dp.length; i++) {
            if(arr[i - 1] >= 0) {
                dp[i][0] = Math.max(arr[i - 1] * dp[i - 1][0], arr[i - 1]);
                dp[i][1] = arr[i - 1] * dp[i - 1][1];
            } else {
                dp[i][0] = Math.max(arr[i - 1] * dp[i - 1][1], arr[i - 1]);
                dp[i][1] = Math.min(dp[i - 1][0] * arr[i - 1], arr[i - 1]);
            }
            max = Math.max(max, dp[i][0]);
        }
        return max;
    }

```

---

### NC98 判断t1树中是否有与t2树拓扑结构完全相同的子树

**描述**

> 给定彼此独立的两棵二叉树，判断 t1 树是否有与 t2 树拓扑结构完全相同的子树。设 t1 树的边集为 E1，t2 树的边集为 E2，若 E2 等于 E1 ，则表示 t1 树和t2 树的拓扑结构完全相同。

**分析**

一种想法是递归，但是比较模糊每个节点都可以试着去比较当前的的节点，如果当前节点值不一样则递归比较左子树和又子树, 需要辅助函数，判断两个node是否相同，递归的进行比较

```java
public boolean isContains (TreeNode root1, TreeNode root2) {
        // write code here
        if(root2 == null)
             return true;
        if(root1 == null && root2 != null)
             return false;
        if(isSame(root1, root2))
            return true;
        else {
            return isContains(root1.left, root2) || isContains(root1.right, root2);
        }
    }
    private boolean isSame(TreeNode root1, TreeNode root2) {
        if(root1 == null && root2 == null)
            return true;
        if(root1 != null && root2 == null || root1 == null && root2 != null)
            return false;
        if(root1.val != root2.val)
            return false;
        return isSame(root1.left, root2.left) && isSame(root1.right, root2.right);
    }

```

---

### NC117 合并二叉树

**描述**

> 已知两颗二叉树，将它们合并成一颗二叉树。合并规则是：都存在的结点，就将结点值加起来，否则空的位置就由另一个树的结点来代替。

**分析**

递归进行合并

```java
    public TreeNode mergeTrees (TreeNode t1, TreeNode t2) {
        // write code here
        if(t1 == null) return t2;
        if(t2 == null) return t1;
        t1.val = t1.val + t2.val;
        t1.left = mergeTrees(t1.left, t2.left);
        t1.right = mergeTrees(t1.right, t2.right);
        return t1;
    }
```

---

### NC135 股票交易的最大收益（二）

**描述**

> 假定你知道某只股票每一天价格的变动。你最多可以同时持有一只股票。但你最多只能进行两次交易（一次买进和一次卖出记为一次交易。买进和卖出均无手续费）。请设计一个函数，计算你所能获得的最大收益。

**分析**

来了来了，这个是两次交易，可以这么思考类似于动态规划吧`dp[i]`表示前`i`个不包括`i`执行一次交易能够获得的最大利润值，然后我们反向计算，计算`arr[i]`在`i`处买进能获得的最大收益，两者的和即为最终的答案

```java
 public int maxProfit (int[] prices) {
        // write code here
        int len = prices.length;
        if (len == 0) return 0;
        int[] dpPre = new int[len];
        int[] dpPost = new int[len];
        int min = prices[0], max = prices[len-1];
        for (int i=1;i<len;++i){
            dpPre[i] = Math.max(dpPre[i-1], prices[i]-min);
            min = Math.min(min, prices[i]);
        }
        
        for (int i=len-2;i>=0;--i){
            dpPost[i] = Math.max(dpPost[i+1], max-prices[i]);
            max = Math.max(max, prices[i]);
        }
        
        int profit = 0;
        for (int i=0;i<len;++i){
            profit = Math.max(profit, dpPre[i]+(i==len-1 ? 0 : dpPost[i+1]));
        }
        return profit;
    }

```

```java
public int maxProfit (int[] prices) {
        // write code here
        if(prices.length==0){
            return 0;
        }
        int firstbuy=prices[0],secondbuy=prices[0];
        int profit1=0,profit2=0;
        for(int i=0;i<prices.length;i++){
            firstbuy=Math.min(firstbuy,prices[i]);
            profit1=Math.max(profit1,prices[i]-firstbuy);
            //为什么第二次价格是这样选
            secondbuy=Math.min(secondbuy,prices[i]-profit1);
            profit2=Math.max(profit2,prices[i]-secondbuy);
        }
        return profit2;
    }

```

----

### NC44 通配符匹配

**描述**

> 请实现支持'?'and'*'.的通配符模式匹配
>
> '?' 可以匹配任何单个字符。
>
> '*' 可以匹配任何字符序列（包括空序列）。
>
> 返回两个字符串是否匹配
>
> 函数声明为：
>
> bool isMatch(const char *s, const char *p)
>
> 下面给出一些样例：
>
> isMatch("aa","a") → false
>
> isMatch("aa","aa") → true
>
> isMatch("aaa","aa") → false
>
> isMatch("aa", "*") → true
>
> isMatch("aa", "a*") → true
>
> isMatch("ab", "?*") → true
>
> isMatch("aab", "d*a*b") → false

**分析**

动态规划，还有一个题目和这个比较类似

```java
 public boolean isMatch(String s, String p) {
        int lp = p.length();
        int ls = s.length();
        boolean[][] dp = new boolean[lp + 1][ls + 1];
        dp[0][0] = true;
        for(int i = 1; i < lp + 1; i++) {
            char chp = p.charAt(i - 1);
            if(chp == '*')
                dp[i][0] = dp[i - 1][0];
            for(int j = 1; j < ls + 1; j++){
                char chs = s.charAt(j - 1);
                if(chp == '*') {
                    dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 1] || dp[i - 1][j];
                } 
                if(chp == '?' || chs == chp) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[lp][ls];   
    }

```

----

### NC77 调整数组顺序使奇数位于偶数前面

**描述**

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**分析**

```java
public int[] reOrderArray (int[] array) {
        // write code here
        int count = 0;
        for(int num : array) 
            if(num % 2 == 1)
                count++;
        if(count == array.length) return array;
        int[] res = new int[array.length];
        int even = count, odd = 0;
        for(int num : array) {
            if(num % 2 == 1) {
                res[odd++] = num;
            } else {
                res[even++] = num;
            }
        }
        return res;
    }

```

-----

### NC129 阶乘末尾0的数量

**描述**

> 给定一个非负整数 N，返回 N! 结果的末尾为 0 的数量。N!是指自然数 N 的阶乘,即:N!=1*2*3…(N-2)*(N-1)*N。

**分析**

这是一个数学题，0是由2 与5相乘产生的，而2的个数远多于5 因此就是求5的个数，比如5！=5 * 4 * 3 * 2 * 1有一个5，因此有一个0，10！=10 * 9 * 1 有两个5 10 / 5， 25！ 一个五的有5 个

```java
   public long thenumberof0 (long n) {
        // write code here
        long count = 0;
        while(n > 0) {
            count += (n / 5);
            n = n / 5;
        }
        return count;
    }
```

---

### NC58 找到搜索二叉树中两个错误的节点

**描述**

> 一棵二叉树原本是搜索二叉树，但是其中有两个节点调换了位置，使得这棵二叉树不再是搜索二叉树，请按升序输出这两个错误节点的值。(每个节点的值各不相同)

**分析**

中序遍历，有两个数的位置在数组中会不太对劲，不过需要考虑多种情况，第一次遇到逆序对时记录两个值，如果再次遇到覆盖第二个值这题看起来很简单，但细想一下操作起来好像又很麻烦，本身是一颗二叉搜索树，那么我按照中序遍历本来应该得到的是一个顺序排列的数组对吧，那就是在这个数组种有两个数字交换了，我们要找出两个数字，是这样一种思路，实现了一种思路，但是感觉有些复杂，那就是这些数字有两种情况，要嘛他们相邻，要嘛他们是数组种的两个奇怪的点，第一次出现逆序的前一个数，以及第二次出现逆序的后一个数

```java
public int[] findError (TreeNode root) {
        // write code here
        if(root == null) return null;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root, pre = null;
        boolean find = false;
        int[] res = new int[2];
        while(cur != null || !stack.isEmpty()) {
            if(cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                if(pre == null) {
                    pre = cur;
                } else {
                    if(cur.val < pre.val) {
                        if(!find) {
                            find = true;
                            res[0] = pre.val;
                            res[1] = cur.val;
                        } else {
                            res[1] = cur.val;
                            Arrays.sort(res);
                            return res;
                        }
                    }
                    pre = cur;
                }
                cur = cur.right;
            }
        }
        Arrays.sort(res);
        return res;
    }

```

-----

### NC142 最长重复子串

**描述**

> 一个重复字符串是由两个相同的字符串首尾拼接而成，例如abcabc便是长度为6的一个重复字符串，而abcba则不存在重复字符串。
>
> 给定一个字符串，请编写一个函数，返回其最长的重复字符子串。若不存在任何重复字符子串，则返回0。

**分析**

硬搜，写一个辅助函数判断某一个偶数长的字符串是否是重复子串，然后对所有可能的子串进行check

```java
public int solve (String a) {
        // write code here
        if (a == null || a.length() <= 1) return 0;
        char[] chars = a.toCharArray();
        int len = chars.length;
        int maxLen = chars.length / 2;
        for (int i = maxLen; i >= 1;--i){
            for (int j = 0; j <= len - 2 * i;++j){
                if (check(chars, j, i)) 
                    return 2 * i;
            }
        }
        return 0;
    }
    public boolean check(char[] chars, int start, int len){
        for (int i = start;i < start + len;++i){
            if (chars[i] != chars[i +len]) 
                  return false;
        }
        return true;
    }

```

----

### NC144 不相邻最大子序列和

**描述**

> 给你一个n（ $1\leq n\leq10^5$），和一个长度为n的数组，在不同时选位置相邻的两个数的基础上，求该序列的最大子序列和（挑选出的子序列可以为空）。

**分析**

动态规划，太标准的动态规划问题了

```java
    public long subsequence (int n, int[] array) {
        // write code here
        if(n == 1) return array[0];
        long[] dp = new long[n];
        dp[0] = array[0];
        dp[1] = Math.max(array[1], array[0]);
        for(int i = 2; i < dp.length; i++) {
            dp[i] = Math.max(array[i] + dp[i - 2], dp[i - 1]);
        }
        return dp[n - 1];
    }
```

----

### NC64 二叉搜索树与双向链表

**描述**

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
>
> 注意:
>
> 1.要求不能创建任何新的结点，只能调整树中结点指针的指向。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继
>
> 2.返回链表中的第一个节点的指针
>
> 3.函数返回的TreeNode，有左右指针，其实可以看成一个双向链表的数据结构
>
> 4.你不用输出或者处理，示例中输出里面的英文，比如"From left to right are:"这样的，程序会根据你的返回值自动打印输出
>
> 示例:
>
> 输入: {10,6,14,4,8,12,16}
>
> 输出:From left to right are:4,6,8,10,12,14,16;From right to left are:16,14,12,10,8,6,4;
>
> 解析:
>
> 输入就是一棵二叉树，如上图，输出的时候会将这个双向链表从左到右输出，以及
>
> 从右到左输出，确保答案的正确a

**分析**

如果允许利用另外的内存，就先中序遍历存起来，然后访问list 将所有的节点根据要求连接起来

```java
public TreeNode Convert(TreeNode pRootOfTree) {
        TreeNode cur = pRootOfTree;
        if(cur == null) return cur;
        Stack<TreeNode> stack = new Stack<>();
        ArrayList<TreeNode> list = new ArrayList<>();
        while(cur != null || !stack.isEmpty()) {
            if(cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                list.add(cur);
                cur = cur.right;
            }
        }
        for(int i = 0; i < list.size() - 1; i++) {
            cur = list.get(i);
            TreeNode next = list.get(i + 1);
            cur.right = next;
            next.left = cur;
        }
        return list.get(0);
    }
}

```

递归方式

```java
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == nullptr) return nullptr;
        TreeNode* pre = nullptr;
         
        convertHelper(pRootOfTree, pre);
         
        TreeNode* res = pRootOfTree;
        while(res ->left)
            res = res ->left;
        return res;
    }
     
    void convertHelper(TreeNode* cur, TreeNode*& pre)
    {
        if(cur == nullptr) return;
         
        convertHelper(cur ->left, pre);
         
        cur ->left = pre;
        if(pre) pre ->right = cur;
        pre = cur;
         
        convertHelper(cur ->right, pre);  
         
    }
};
```

----

### NC84 完全二叉树结点数

**描述**

> 给定一棵完全二叉树的头节点head，返回这棵树的节点个数。如果完全二叉树的节点数为N，请实现时间复杂度低于O(N)的解法。

**分析**

完全二叉树还是有一些特性的比如i与2i节点之间的关系

```java
    public int nodeNum(TreeNode head) {
        if(head == null) return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(head);
        int count = 0;
        while(!queue.isEmpty()) {
            count++;
            TreeNode cur = queue.poll();
            if(cur.left != null) queue.offer(cur.left);
            if(cur.right != null) queue.offer(cur.right);
        }
        return count;
    }
```

完全二叉树 是指最后一层或者 倒数第二层的左边是满的，右边允许为空值，那么 考虑下来确实可以提前终止遍历，当发现某个节点的左或者右为空时，可以考虑计算了，利用二叉树性质可以提前一点点进行终止，但是帮助有限

```java
    public int nodeNum(TreeNode head) {
        if(head == null) return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(head);
        int count = 0;
        while(!queue.isEmpty()) {
            count++;
            TreeNode cur = queue.poll();
            if(cur.left != null) queue.offer(cur.left);
            else return count * 2 - 1;
            if(cur.right != null) queue.offer(cur.right);
            else return count * 2;
        }
        return count;
    }
```

```c++
public:
    int get_height(struct TreeNode *head){
        return head == NULL ? 0 : get_height(head->left) + 1;
    }
    int nodeNum(struct TreeNode* head) {
        if(head == NULL)
            return 0;
        int height = get_height(head);
        int cur_h = 1;
        if(head->right != NULL){
            struct TreeNode *new_head = head->right;
            cur_h++;
            while(new_head->left != NULL){
                new_head = new_head->left;
                cur_h++;
            }
        }
        if(cur_h == height)
            return pow(2, height-1) + nodeNum(head->right);
        else
            return pow(2, cur_h-1) + nodeNum(head->left);
    }
};
```

----

### NC125 未排序数组中累加和为给定值的最长子数组长度

**描述**

> 给定一个无序数组arr, 其中元素可正、可负、可0。给定一个整数k，求arr所有子数组中累加和为k的最长子数组长度

**分析**

求子数组长度看来是不允许排序了，有一个$O(n^2)$的兜底算法，任何一个数开始计算任意长度的连续子数组看是否等于k，所以最好的解法需要突破这个瓶颈。 一般和为定植，或者在数组种寻找都会利用两个指针进行判断，有许多类似的题目如两个数和为定植，利用二分查找或者hash表，三个数类似，这里是求最长，这里又个条件子数组，那必定是连续的，可以考虑求类和，这样可以求得所有子数组的和，n*n的复杂度, 解析上用了hashmap。 

这个题比较巧妙，利用连续的子数组的性质，用一个hashmap来存储，hashmap 种存的东西是针对某个连续子数组和sum 它最早出现的最后一个节点的序号

```java
public int maxlenEqualK (int[] arr, int k) {
        if(arr == null) return 0;
        int sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for(int i = 0; i < arr.length; i++) {
            sum += arr[i];
            int delta = sum - k;
            if(map.containsKey(sum - k)) res = Math.max(res, i - map.get(delta));
            if(!map.containsKey(sum)) map.put(sum, i);
        }
        return res;
    }

```

```java
    public int maxlenEqualK (int[] arr, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0, result = 0;
        for(int i = 0; i < arr.length; i++) {
            sum += arr[i];
            int delta = sum - k;
            if(sum == k) {
                result = i + 1;
            }
            if(map.containsKey(delta)){
                result = Math.max(i - map.get(delta), result);
            } else if(!map.containsKey(sum))
                map.put(sum, i);
        }
        return result;
    }
```

----



### NC31 第一个只出现一次的字符

**描述**

> 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

```java
    public int FirstNotRepeatingChar(String str) {
        Map<Character, Integer> map = new HashMap<>();
        for(char ch : str.toCharArray()) {
            if(map.containsKey(ch)) {
                map.put(ch, map.get(ch) + 1);
            }else {
                map.put(ch, 1);
            }
        }
        char[] chs = str.toCharArray();
        for(int i = 0; i < chs.length; i++) {
            if(map.get(chs[i]) == 1)
                return i;
        }
        return -1;
    }
```

---

### NC23 划分链表

**描述**

> 给出一个链表和一个值 
>
> ，以 
>
> 为参照将链表划分成两部分，使所有小于 
>
> 的节点都位于大于或等于 
>
> 的节点之前。
>
> 两个部分之内的节点之间要保持的原始相对顺序。
>
> 例如：
>
> 给出 $1\to 4 \to 3 \to 2 \to 5 \to 2$ x=3,
>
> 返回 $1\to 2 \to 2 \to 4 \to 3 \to 5$.

**分析**

理解题意最关键啊

```java
public ListNode partition (ListNode head, int x) {
        // write code here
        if(head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode fl = dummy, curX = head, preX = dummy;
        while(fl.next != null && fl.next.val < x)
            fl = fl.next;
        while(curX != null && curX.val < x)
            curX = curX.next;
        while(curX != null) {
            if(curX.val < x) {
                preX.next = curX.next;
                curX.next = fl.next;
                fl.next = curX;
                fl = curX;
                curX = preX.next;
            } else {
                preX = curX;
                curX = preX.next;
            }
        }
        return dummy.next;
    }

```

-----

### NC75 数组中只出现一次的两个数字

**描述**

> 一个整型数组里除了两个数字只出现一次，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

**分析**

如果允许内存肯定是统计，这道题有特殊的解法

```java
 public int[] FindNumsAppearOnce (int[] array) {
        int num = 1;
        for(int n : array)
            num ^= n;
        num ^= 1;
        int idx = 0;
        while(idx < 32 && ((num >> idx) & 1) == 0) idx++;
        int num1 = 1, num2 = 1;
        for(int n : array) {
            if(((n >> idx) & 1) > 0) {
                num1 ^= n;
            } else {
                num2 ^= n;
            }
        }
        num1 ^= 1; num2 ^= 1;
        int[] res = {num1, num2};
        Arrays.sort(res);
        return res;
    }


```

----

### NC13 **二叉树的最大深度**

**描述**

> 求给定二叉树的最大深度，
>
> 最大深度是指树的根结点到最远叶子结点的最长路径上结点的数量。

**分析**

这个是深度，从根节点出发，比较好计算，递归最为简单，也可以利用层序遍历统计最远的层数

```java
 public int maxDepth (TreeNode root) {
        // write code here
        return nonRec(root);
    }
    
    private int rec(TreeNode root) {
        if(root == null)
            return 0;
        return Math.max(rec(root.left), rec(root.right)) + 1;
    }
    private int nonRec(TreeNode root) {
        if(root == null)
            return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int count = 0;
        while(!queue.isEmpty()) {
            count++;
            int size = queue.size();
            for(int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if(cur.left != null)
                    queue.add(cur.left);
                if(cur.right != null)
                    queue.add(cur.right);
            }
        }
        return count;
    }
```

----

### **NC63** **扑克牌顺子**

**描述**

> 现在有2副扑克牌，从扑克牌中随机五张扑克牌，我们需要来判断一下是不是顺子。
> 有如下规则：
> \1. A为1，J为11，Q为12，K为13，A不能视为14
> \2. 大、小王为 0，0可以看作任意牌
> \3. 如果给出的五张牌能组成顺子（即这五张牌是连续的）就输出true，否则就输出false。
> 例如：给出数据[6,0,2,0,4]
> 中间的两个0一个看作3，一个看作5 。即：[6,3,2,5,4]
> 这样这五张牌在[2,6]区间连续，输出true
> 数据保证每组5个数字，每组最多含有4个零，数组的数取值为 [0, 13]

**分析**

有很多种思考的方式，一种想法就是排除法，哪些不可能成为顺子，两副牌有4个王，如果排序后两者相差超过4肯定没戏，如果两者是一样的肯定没戏，如果超过的个数大于王的个数也没戏，足够则减去王的个数，就是这么简单吧

```java
    public boolean IsContinuous(int [] numbers) {
        Arrays.sort(numbers);
        List<Integer> cards = new ArrayList<>();
        int jokers = 0;
        for(int n : numbers) {
            if(n == 0)
                jokers++;
            else
                cards.add(n);
        }
        for(int i = 1; i < cards.size(); i++) {
            int cur = cards.get(i);
            int pre = cards.get(i - 1);
            if(cur == pre || cur - pre - 1 > jokers)
                return false;
            if(cur == pre + 1)
                continue;
            jokers -= (cur - pre - 1);
        }
        return true;
    }
```

或者就是直接统计需要多少个王，然后比较与jokers的大小关系

```java
    private boolean test(int[] numbers) {
                int count = 0;
        ArrayList<Integer> list = new ArrayList<>();
        for(int n : numbers)
            if(n > 0) list.add(n);
                else  count++;
        Collections.sort(list);
        int cntNeed = 0;
        for(int i = 1; i < list.size(); i++) {
            int cur = list.get(i);
            int pre = list.get(i - 1);
            if(cur == pre) return false;
            cntNeed += (cur - pre  - 1);
        }
        if(cntNeed > count) return false;
        return true;
    }
```

---

### **NC113** **验证IP地址**

**描述**

> 编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址
>
> IPv4 地址由十进制数和点来表示，每个地址包含4个十进制数，其范围为 0 - 255， 用(".")分割。比如，172.16.254.1；
> 同时，IPv4 地址内的数不会以 0 开头。比如，地址 172.16.254.01 是不合法的。
>
> IPv6 地址由8组16进制的数字来表示，每组表示 16 比特。这些组数字通过 (":")分割。比如, 2001:0db8:85a3:0000:0000:8a2e:0370:7334 是一个有效的地址。而且，我们可以加入一些以 0 开头的数字，字母可以使用大写，也可以是小写。所以， 2001:db8:85a3:0:0:8A2E:0370:7334 也是一个有效的 IPv6 address地址 (即，忽略 0 开头，忽略大小写)。
>
> 然而，我们不能因为某个组的值为 0，而使用一个空的组，以至于出现 (::) 的情况。 比如， 2001:0db8:85a3::8A2E:0370:7334 是无效的 IPv6 地址。
> 同时，在 IPv6 地址中，多余的 0 也是不被允许的。比如， 02001:0db8:85a3:0000:0000:8a2e:0370:7334 是无效的。
>
> 说明: 你可以认为给定的字符串里没有空格或者其他特殊字符。

**分析**

将IPV4和IPV6分析判断写两个函数，题目中已经把二者的判断条件写的比较清楚了，依次编写

```java
    public String solve (String IP) {
        // write code here
       if(isIPV4(IP))
           return "IPv4";
        else if(isIPV6(IP))
            return "IPv6";
        return "Neither";
    }
    
    
    private boolean isIPV4(String IP) {
        String[] ips = IP.split("\\.");
        if(ips.length != 4)
            return false;
        for(String ip : ips) {
            if(ip == null || (ip.charAt(0) == '0') && ip.length() > 1)
                return false;
            try {
                int count = Integer.parseInt(ip);
                if(count < 0 || count > 255)
                    return false;
            } catch(Exception e) {
                return false;
            }
        }
        return true;
    }
    
    private boolean isIPV6(String IP) {
        String[] ips = IP.split(":");
        if(ips.length != 8)
            return false;
        for(String ip : ips) {
            if(ip == null || ip.length() > 4)
                return false;
            for(char s : ip.toCharArray()) {
                if((s >= '0') && (s <= '9') || ((s >= 'a') && (s <= 'z') ) || ((s >= 'A') && (s <= 'Z')))
                    continue;
                else return false;
            }
        }
        return true;
    }
```

---

### NC46 加起来和为目标值的组合

**描述**

> 给出一组候选数*C* 和一个目标数*T*，找出候选数中起来和等于 *T* 的所有组合。*C* 中的每个数字在一个组合中只能使用一次。
>
> 注意：
>
> - 题目中所有的数字（包括目标数*T* ）都是正整数
> - 组合中的数字 $(a_1, a_2, … , a_k)$ 要按非递增排序 $(a_1 \leq a_2 \leq … \leq a_k)$.
> - 结果中不能包含重复的组合
> - 组合之间的排序按照索引从小到大依次比较，小的排在前面，如果索引相同的情况下数值相同，则比较下一个索引。

**分析**

需要递归查找，或者说回溯, 稍微注意一下去重的条件，就是在一个回溯中相同的值只选取一个

```java
    public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        helper(num, 0, target, new ArrayList<Integer>(), result);
        return result;
    }
    
    private void helper(int[] num, int idx, int target, ArrayList<Integer> temp, ArrayList<ArrayList<Integer>> result) {
        if(target == 0) {
            result.add(new ArrayList<Integer>(temp));
            return;
        }
        for(int i = idx; i < num.length; i++) {
            if(i > idx && num[i] == num[i - 1])
                continue;
            if(target >= num[i]) {
                temp.add(num[i]);
                helper(num, i + 1, target - num[i], temp, result);
                temp.remove(temp.size() - 1);
            }
        }
    } 
```

---

### **NC73** **数组中出现次数超过一半的数字**

**描述**

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组[1,2,3,2,2,2,5,4,2]。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。你可以假设数组是非空的，并且给定的数组总是存在多数元素。1<=数组长度<=50000

**分析**

超过一半，找中位数肯定是可以的，不考虑内存的话统计个数也是很直接的想法

```java
   public int MoreThanHalfNum_Solution(int [] array) {
         int len = array.length;
         Map<Integer, Integer> map = new HashMap<>();
         for(int n : array) {
             if(map.containsKey(n)) {
                 map.put(n, map.get(n) + 1);
             } else {
                 map.put(n, 1);
             }
             if(map.get(n) > len / 2)
                 return n;
         }
         return -1;
     }
```

利用times， 大于一半的数就算是最差情况这样子依然是可以筛选出来的

```java
     public int MoreThanHalfNum_Solution(int [] array) {
         int times = 0;
         int result = -1;
         for(int n : array) {
             if(times == 0){
                 result = n;
                 times = 1;
             } else {
                 if(n == result) times++;
                 else times--;
             }
         }
         if(result == -1) return 0;
         return result;
     }
```

----

### **NC157** **单调栈**

**描述**

> 给定一个可能含有重复值的数组 arr，找到每一个 i 位置左边和右边离 i 位置最近且值比 arr[i] 小的位置。返回所有位置相应的信息。位置信息包括：两个数字 L 和 R，如果不存在，则值为 -1，下标从 0 开始。

**分析**

我感觉这个题目的名字已经把做法给暴露了，就是在栈中维护一个单调递增的序列，然后给出答案，从左往右左两次

```java
public int[][] foundMonotoneStack (int[] nums) {
        // write code here
        int[][] result = new int[nums.length][2];
        helper(result, nums, 0, 1, 0, nums.length);
        helper(result, nums, 1, -1, nums.length -1, -1);
        return result;
    }
    
    private void helper(int[][] result, int[] nums, int idx, int delta, int start, int end) {
        Stack<Integer> stack = new Stack<>();
        for(int i = start; i != end;) {
            if(stack.isEmpty()) {
                stack.push(i);
                result[i][idx] = -1;
                i+=delta;
            } else {
                if(nums[i] > nums[stack.peek()]) {
                    result[i][idx] = stack.peek();
                    stack.push(i);
                    i+=delta;
                } else if(nums[i] == nums[stack.peek()]) {
                    result[i][idx] = result[stack.peek()][idx];
                    i+=delta;
                } else{
                    while(!stack.isEmpty() && nums[i] < nums[stack.peek()]) {
                        stack.pop();
                    }
                }
            }
        }
    }
```

----

### **NC90** **包含min函数的栈**

**描述**

> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数，并且调用 min函数、push函数 及 pop函数 的时间复杂度都是 O(1)
>
> push(value):将value压入栈中
>
> pop():弹出栈顶元素
>
> top():获取栈顶元素
>
> min():获取栈中最小元素
>
> 示例:
>
> 输入:  ["PSH-1","PSH2","MIN","TOP","POP","PSH1","TOP","MIN"]
>
> 输出:  -1,2,1,-1
>
> 解析:
>
> "PSH-1"表示将-1压入栈中，栈中元素为-1
>
> "PSH2"表示将2压入栈中，栈中元素为2，-1
>
> “MIN”表示获取此时栈中最小元素==>返回-1
>
> "TOP"表示获取栈顶元素==>返回2
>
> "POP"表示弹出栈顶元素，弹出2，栈中元素为-1
>
> "PSH-1"表示将1压入栈中，栈中元素为1，-1
>
> "TOP"表示获取栈顶元素==>返回1
>
> “MIN”表示获取此时栈中最小元素==>返回-1

**分析**

记录min与当前数值的差值，进行计算

```java
public class Solution {

    private int min = Integer.MAX_VALUE;
    Stack<Integer> stack = new Stack<>();
    public void push(int node) {
        if(stack.isEmpty()) {
            min = node;
            stack.push(node - min);
        } else {
            stack.push(node - min);
            if(node < min) {
                min = node;
            }
        }
    }
    
    public void pop() {
       int val = stack.pop();
       if(val <= 0) {
           min = min - val;
       }
    }
    
    public int top() {
        int val = stack.peek();
        if(val <= 0) return min;
        else return min + val;
    }
    
    public int min() {
        return min;
    }
     }
```

----

### NC51 合并k个一排序的链表

> 合并\ k *k* 个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度。

**分析**

每次移动最小值所对应的链表头，利用优先级队列，每次获取最小的节点然后移动构建新的链表

```java
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        ListNode dummy = new ListNode(-1), cur = dummy;
        PriorityQueue<ListNode> queue = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for(ListNode node : lists) {
            if(node != null) {
                queue.add(node);
            }
        }
        while(!queue.isEmpty()) {
            ListNode tmp = queue.poll();
            cur.next = tmp;
            cur = cur.next;
            tmp = tmp.next;
            if(tmp != null)
                queue.add(tmp);
        }
        return dummy.next;
    }
```

------

### NC121  **字符串的排列**

**描述**

> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

**分析**

排列问题是要基于递归求解的，计算排列

```java
    public ArrayList<String> Permutation(String str) {
        char[] chs = str.toCharArray();
        Arrays.sort(chs);
        ArrayList<String> result = new ArrayList<>();
        boolean[] isVisisted = new boolean[chs.length];
        helper(chs, "", isVisisted, result);
        return result;
    }
    
    private void helper(char[] chs, String temp, boolean[] isVisted, ArrayList<String> result) {
        if(temp.length() == chs.length) {
            if(!result.contains(temp))
                result.add(temp);
            return;
        }
        for(int i = 0; i < chs.length; i++) {
            if(!isVisted[i]) {
                isVisted[i] = true;
                helper(chs, temp + chs[i], isVisted, result);
                isVisted[i] = false;
            }
        }
    }
```

第二种思考方式，遍历每个位置，构造出排列，每个位置i之后的任何一个字符都能在此位置上填充

```java
    public ArrayList<String> Permutation(String str) {
        char[] chs = str.toCharArray();
        Arrays.sort(chs);
        ArrayList<String> result = new ArrayList<>();
        perHelper(chs, 0, result);
        Collections.sort(result);
        return result;
    }
    
    private void perHelper(char[] chs, int idx, ArrayList<String> result) {
        if(idx == chs.length - 1) {
            String tmp = String.valueOf(chs);
            if(!result.contains(tmp))
                result.add(tmp);
            return;
        }
        for(int i = idx; i < chs.length; i++) {
            swap(chs, idx, i);
            perHelper(chs, idx + 1, result);
            swap(chs, idx, i);
        }
    }
```

----

### NC109 岛屿数量

**描述**

> 给一个01矩阵，1代表是陆地，0代表海洋， 如果两个1相邻，那么这两个1属于同一个岛。我们只考虑上下左右为相邻。
>
> 岛屿: 相邻陆地可以组成一个岛屿（相邻:上下左右） 判断岛屿个数。

**分析**

深度优先遍历，考察的地方应该在于怎么把这个深度优先给写的简洁全面

```java
    public int solve (char[][] grid) {
        int count = 0;
        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[i].length; j++) {
                if(grid[i][j] == '1') {
                    count++;
                    helper(grid, i, j);
                }
            }
        }
        return count;
    }
    
    private void helper(char[][] grid, int x, int y) {
      //  if((x >= 0 && x < grid.length) && (y >= 0 && y < grid[0].length)){
            grid[x][y] = '0';
        //}
        if(x > 0 && grid[x - 1][y] == '1')
            helper(grid, x - 1, y);
        if(x < grid.length - 1 && grid[x + 1][y] == '1')
            helper(grid, x + 1, y);
        if(y > 0 && grid[x][y - 1] == '1')
            helper(grid, x, y - 1);
        if(y < grid[0].length - 1 && grid[x][y + 1] == '1')
            helper(grid, x, y + 1);
    }
```

-----

###NC70 单链表的排序

**描述**

给定一个无序单链表，实现单链表的排序(按升序排序)。

**分析**

1，最简单直接的应该是插入排序

```java
    private ListNode insertSort(ListNode head) {
        if(head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(-1), curTail = head, curHead = head.next;
        dummy.next = curTail;
        while(curHead != null) {
            if(curHead.val >= curTail.val) {
                curTail = curHead;
            } else {
                ListNode pre = dummy, cur = dummy.next;
                curTail.next = curHead.next;
                while(cur.val < curHead.val) {
                    pre = cur;
                    cur = pre.next;
                }
                curHead.next = cur;
                pre.next = curHead;
            }
            curHead = curTail.next;
        }
        return dummy.next;
    }
```

也可以做一个mergeSort,partition的时候注意两个节点的情况

```java
private ListNode mergeSort(ListNode head) {
        if(head == null || head.next == null)
            return head;
        ListNode slow = head, fast = head.next;
        while(fast.next != null) {
            slow = slow.next;
            fast = fast.next;
            if(fast.next != null)
                fast = fast.next;
        }
        fast = slow.next;
        slow.next = null;
        ListNode first = mergeSort(head);
        ListNode second = mergeSort(fast);
        ListNode dummy = new ListNode(-1), cur = dummy;
        while(first != null && second != null) {
            if(first.val < second.val){
                cur.next = first;
                first = first.next;
            } else {
                cur.next = second;
                second = second.next;
            }
            cur = cur.next;
        }
        while(first != null) {
            cur.next = first;
            cur = cur.next;
            first = first.next;
        }
        while(second != null) {
            cur.next = second;
            cur = cur.next;
            second = second.next;
        }
        return dummy.next;
    }
```

----

### NC59 矩阵的最小路径和

**描述**

> 给定一个 n * m 的矩阵 a，从左上角开始每次只能向右或者向下走，最后到达右下角的位置，路径上所有的数字累加起来就是路径和，输出所有的路径中最小的路径和。

**分析**

典型动态规划问题啊

```java
    public int minPathSum (int[][] matrix) {
        if(matrix == null)
            return 0;
       int h = matrix.length, w = matrix[0].length;
       int[][] dp = new int[h][w];
       dp[0][0] = matrix[0][0];
       for(int i = 1; i < dp.length; i++)
           dp[i][0] = dp[i - 1][0] + matrix[i][0];
       for(int i = 1; i < dp[0].length; i++)
           dp[0][i] = dp[0][i - 1] + matrix[0][i];
       for(int i = 1; i < dp.length; i++) {
           for(int j = 1; j < dp[i].length; j++) {
               dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
           }
       }
       return dp[h - 1][w - 1];
    }
```

-----

### NC62 平衡二叉树

**描述**

> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
>
> 在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树
>
> **平衡二叉树**（Balanced Binary Tree），具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

**分析**

这个题目可以利用高度来做文章，递归求解

```java
    public boolean IsBalanced_Solution(TreeNode root) {
         height(root);
        return isBalance;
    }
    
    private int height(TreeNode root) {
        if(root == null)
            return 0;
        int l = height(root.left), r = height(root.right);
        if(Math.abs(l - r) > 1)
            isBalance = false;
        return Math.max(l, r) + 1;
    }
```

----

### NC96 判断一个链表是否为回文结构

**分析**

两种做法，一是利用栈，二是反转链表

```java
private boolean stackWay(ListNode head) {
        Stack<ListNode> stack = new Stack<>();
        ListNode cur = head;
        while(cur != null) {
            stack.push(cur);
            cur = cur.next;
        }
        cur = head;
        while(cur != null) {
            if(cur.val != stack.peek().val)
                return false;
            stack.pop();
            cur = cur.next;
        }
        return true;
    }
```

```java
    private boolean reverseWay(ListNode head) {
        if(head == null || head.next == null) return true;
        ListNode dummy = new ListNode(-1), cur = head;
        while(cur != null) {
            ListNode temp = new ListNode(cur.val);
            temp.next = dummy.next;
            dummy.next = temp;
            cur = cur.next;
        }
        ListNode ncur = dummy.next;
        cur = head;
        while(cur != null) {
            if(cur.val != ncur.val)
                return false;
            cur = cur.next;
            ncur = ncur.next;
        }
        return true;
    }
```

----

### NC35 最小编辑代价

**描述**

> 给定两个字符串str1和str2，再给定三个整数ic，dc和rc，分别代表插入、删除和替换一个字符的代价，请输出将str1编辑成str2的最小代价。

**分析**

动态规划的问题主要是判断初值与转移方程

```java
    public int minEditCost (String str1, String str2, int ic, int dc, int rc) {
        int l1 = str1.length(), l2 = str2.length();
        int[][] dp = new int[l1 + 1][l2 + 1];
        for(int i = 1; i < dp.length; i++)
           dp[i][0] = dc * i;
        for(int i = 1; i < l2 + 1; i++)
           dp[0][i] = ic * i;
        for(int i = 1; i < dp.length; i++) {
            for(int j = 1; j < dp[i].length; j++) {
                char s1 = str1.charAt(i - 1);
                char s2 = str2.charAt(j - 1);
                if(s1 == s2) {
                  // 相等最完美
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                  // 不想等的话 一次考虑替换，删除，插入这三种选择选取最小代价哦
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + rc, Math.min(dp[i - 1][j] + dc, dp[i][j - 1] + ic));
                }
            }
        }
        return dp[l1][l2];
    }
```

----

### NC5 **二叉树根节点到叶子节点的所有路径和**

**描述**

> 给定一个仅包含数字\ 0-9 0−9 的二叉树，每一条从根节点到叶子节点的路径都可以用一个数字表示。
> 例如根节点到叶子节点的一条路径是1\to 2\to 31→2→3,那么这条路径就用\ 123 123 来代替。
> 找出根节点到叶子节点的所有路径表示的数字之和
> 例如：
>
> ![img](https://uploadfiles.nowcoder.com/images/20200807/999991351_1596786228797_BC85E8592A231E74E5338EBA1CFB2D20)
>
> 这颗二叉树一共有两条路径，
> 根节点到叶子节点的路径 1\to 21→2 用数字\ 12 12 代替
> 根节点到叶子节点的路径 1\to 31→3 用数字\ 13 13 代替
> 所以答案为\ 12+13=25 12+13=25

**分析**

首先一个注意要点是叶子结点的定义，那就意味着此节点的左右孩子为空

```java
     int sum = 0;
     public int sumNumbers (TreeNode root) {
         if(root == null)
             return 0;
         helper(root, 0);
         return sum;
     }
    
     private void helper(TreeNode root, int pre) {
         int curVal = pre * 10 + root.val;
         if(root.left == null && root.right == null){
             sum += curVal;
         } else {
            if(root.left != null)  helper(root.left, curVal);
            if(root.right != null) helper(root.right, curVal);
         }
     }
```

---

### NC8 **二叉树根节点到叶子节点和为指定值的路径**

**描述**

> 给定一个二叉树和一个值\ sum *s**u**m*，请找出所有的根节点到叶子节点的节点值之和等于\ sum *s**u**m* 的路径

**分析**

同样的叶子结点的分析，写代码有时候很重要的一部分，不变式的保证，比如我们想让递归的节点永远不是null那么我们在递归之前就要先做一些保证以此维持这个不变式

```java
    public ArrayList<ArrayList<Integer>> pathSum (TreeNode root, int sum) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        helper(root, sum, new ArrayList<Integer>(), result);
        return result;
    }
    
    private void helper(TreeNode root, int target, ArrayList<Integer> temp, ArrayList<ArrayList<Integer>> result) {
        temp.add(root.val);
        if(root.left == null && root.right == null && root.val == target){
            result.add(new ArrayList<Integer>(temp));
        } else {
            if(root.left != null)  helper(root.left, target - root.val, temp, result);
            if(root.right != null) helper(root.right, target - root.val, temp, result);
        }
        temp.remove(temp.size() - 1);
    }
```

----

### **NC21** **链表内指定区间反转**

**描述**

> 将一个链表\ m *m* 位置到\ n *n* 位置之间的区间反转，要求时间复杂度 O(n)*O*(*n*)，空间复杂度 O(1)*O*(1)。
> 例如：
> 给出的链表为 1\to 2 \to 3 \to 4 \to 5 \to NULL1→2→3→4→5→*N**U**L**L*, m=2,n=4*m*=2,*n*=4,
> 返回 1\to 4\to 3\to 2\to 5\to NULL1→4→3→2→5→*N**U**L**L*.
> 注意：
> 给出的 m*m*,n*n* 满足以下条件：
> 1 \leq m \leq n \leq 链表长度1≤*m*≤*n*≤链表长度

**分析**

反转这种链表的时候有一个不变量，就是原来的head会成为最终的末尾节点这道题值的二刷，先找到m的前驱节点，然后从m的后一个节点开始插入，同时保证m节点每次都挂在他的next.next节点上

```java
public ListNode reverseBetween (ListNode head, int m, int n) {
        ListNode dummy = new ListNode(-1), pre = dummy;
        dummy.next = head;
        for(int i = 1; i < m; i++)
            pre = pre.next;
        head = pre.next;
        ListNode next;
        for(int i = m; i < n; i++) {
            next = head.next;
            head.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return dummy.next;
    }
```

