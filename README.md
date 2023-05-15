# Anagraphs Solver

Solver for a Steam game [Anagraphs: An Anagram Game With a Twist](https://store.steampowered.com/app/1654280/Anagraphs_An_Anagram_Game_With_a_Twist/)

It was used to write walkthrough guide for this game. [Walkthrough Guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2972777433)

## Requirements

-   Python

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python main.py output.txt
```

## Algorithm

### Minimum path between two words

In this game there are two operations that can be done on a word:

1. Move a letter to a different place<br>
   **Example:** `leaps -> lapse` (moving `e` to the end of the word)<br>
2. Rotate a letter by 180 degrees<br>
   **Example:** `reedy -> ready` (rotating second `e` so it becomes `a`)<br>
   All possible rotations are:
    - `a <-> e`
    - `b <-> q`
    - `d <-> p`
    - `h <-> y`
    - `m <-> w`
    - `n <-> u`
    - `j <-> r`

By word here we mean a string of letters, not a word in a dictionary.

Easiest algorithm you can think of to find minimum path between two words is BFS algorithm. It's complexity would be equal to $O(m^{2answer})$, where $m$ is the length of the word and $answer$ is the distance between two words. This is because there are at most $m$ possible rotations and $m$ possible letters to move to $m - 1$ possible locations, thus making at most $m + m(m - 1) = m^2$ neighbours for each word. For quite far words of length 8 (levels 26+) it becomes impractical.

#### Proposed algorithm

1. Split path from one word to another into two pieces:

    1. Path consisting of only rotations from first word to a word that has same letters as second word, but in different order<br>
    2. Path from that word to second word with only letter moves

    **Example:** we want to find path from `leeds` to `lapse`<br>
    First part would be `leeds -> leads -> leaps`<br>
    Second part would be `leaps -> lapse`

2. Finding minimum path for second part can be done by understanding that each letter should be moved at most once and all the letters that don't move is a common subsequence beetween the two words. Thus we can find longest common subsequence (LCS) and then efficiently find path beetween two words. This can be done with dynamic programming in $O(m^2)$ time.
3. We will do step 2 for each possible output of the first part path and thus will find minimum path for the whole path. Number of possible outputs of the first part path is not large for real words. It is equal to production of all commbinations of different groups of letters where you select from number of letters in each group a number of particular letter in that group.<br>
   **Example:** from `leeds` to `lapse`<br>
   We have two groups: `a <-> e` and `d <-> p`<br>
   In the first group we have 2 letters in the word and in the end we have only one `a` (and one `e`) so it contriutes $C^2_1 = 2$<br>
   In the second group we have 1 letter in the word and in the end we have only one `p` so it contributes $C^1_1 = 1$<br>
   Thus total number of combinations equals to $C^2_1 * C^1_1 = 2 * 1 = 2$ and these are `leaps` and `laeps`

#### Personal comments on proposed algorithm

Though proposed algorithm is enough to solve all the game's levels I think it's possible to improve it by modifying longest common subsequence (LCS) algorithm and accounting for the fact that you need to rotate some of the letters (effectively merging first and second parts together), though don't have this algorithm in mind yet.

### Minimum chain from start word that goes through all words

This is the Travelling Salesman Problem without return.

Easiest algorithm you can think of is checking all possible permutations of words and finding the one with minimum cost. This algorithm has complexity $O(n!)$ where $n$ is the number of words. It becomes impractical for levels with lots of words in them (level 10 has 13 words).

#### Proposed algorithm

You can use dynamic programming algorithm, specifically Held–Karp algorithm that solves the problem in time $O(n^{2}2^{n})$

### Minimum chain that goes through all words and has minimum moves according to the game

This game has a bug with counting moves. If you move a letter to the same place it was before it gives you back a move but doesn't forget words you've already visited. Thus you can go through all words in a level except for one (the closest to the starting word), reverse all moves and then do a small amount of moves to get to the last word. One of the problems is that in order to get maximum stars on some levels you actually needs to use this trick.<br>
**Example:** Level 6<br>
In order to get 3 stars you need to solve it in 7 moves or less.<br>
Optimal path is 8 moves long: `ostp -> post -> spot -> pots -> dots -> pots -> opts -> tops -> stop`<br>
Instead you can make game think that you've solved it in 1 move by doing this path:
`ostp -> stop -> tops -> opts -> pots -> dots -> pots -> spot -> pots -> dots -> pots -> opts -> tops -> stop -> ostp -> post`

If you want not only to have minimum amount of moves but also want to have minimum path that achieves it then this problem can be reduced to Steiner tree problem. A good video about possible approaches to this problem are presented in this video [Algorithms Live! - Episode 7 - Steiner Trees](https://www.youtube.com/watch?v=BG4vAoV5kWw)

#### Proposed algorithm

We don't try to solve Steiner tree problem here though think it's quite an interesting thing to do

## Related links
1. [Held-Karp algorithm](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
2. [Algorithms Live! - Episode 7 - Steiner Trees](https://www.youtube.com/watch?v=BG4vAoV5kWw)
