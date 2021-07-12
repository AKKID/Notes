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

