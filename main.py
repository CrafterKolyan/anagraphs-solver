import itertools
import queue
import more_itertools
from collections import defaultdict
from tqdm.auto import tqdm
from functools import lru_cache
import sys


def test(f, expected, *args, **kwargs):
    actual = f(*args, **kwargs)
    if expected != actual:
        pretty_args = []
        if args:
            pretty_args.append(", ".join(map(repr, args)))
        if kwargs:
            pretty_args.append(", ".join([f"{key}={value!r}" for key, value in kwargs.items()]))
        pretty_args = ", ".join(pretty_args)
        print(f"{f.__name__}({pretty_args}) != {expected}. Actual: {actual}")


class Solver:
    def __init__(self, words_filename=None, transpositions=None):
        if transpositions is None:
            # Letters rotated look similar to each other
            transpositions = [("a", "e"), ("m", "w"), ("h", "y"), ("n", "u"), ("p", "d"), ("b", "q"), ("r", "j")]
        mapping = {}
        for a, b in transpositions:
            mapping[a] = b
            mapping[b] = a

        self.transpositions = mapping

        if words_filename is not None:
            with open(words_filename, "r", encoding="utf-8") as f:
                words = [x.strip() for x in f.readlines()]
            words = [x for x in words if x]
            grouped_words = defaultdict(list)
            for word in words:
                group = self.group_of_word(word.strip())
                grouped_words[group].append(word)
            self.words = grouped_words

    def group_of_word(self, word):
        word = [x if x not in self.transpositions or ord(self.transpositions[x]) > ord(x) else self.transpositions[x] for x in word]
        word = "".join(sorted(word))
        return word

    def possible_words(self, word):
        return self.words[self.group_of_word(word)]

    def possible_positions_in_1_move(self, word):
        possible_positions = []
        for i in range(len(word)):
            current_word = word[:i] + word[i + 1 :]
            for j in range(len(word)):
                if i == j:
                    continue
                possible_positions.append(current_word[:j] + word[i] + current_word[j:])

            transposed_letter = self.transpositions.get(word[i])
            if transposed_letter is not None:
                possible_positions.append(word[:i] + transposed_letter + word[i + 1 :])
        return set(possible_positions)

    def in_1_move(self, from_, to):
        return to in self.possible_positions_in_1_move(from_)

    def in_n_moves(self, from_, to, n):
        if n == 0:
            return from_ == to
        if n == 1:
            return self.in_1_move(from_, to)
        for possible_position in self.possible_positions_in_1_move(from_):
            if self.in_n_moves(possible_position, to, n - 1):
                return True
        return False

    def in_n_moves_bfs(self, from_, to, n):
        if n == 0:
            return from_ == to
        if n == 1:
            return self.in_1_move(from_, to)
        visited = set()
        queue = [from_]
        for _ in range(n):
            next_queue = []
            for word in queue:
                for possible_position in self.possible_positions_in_1_move(word):
                    if possible_position == to:
                        return True
                    if possible_position not in visited:
                        visited.add(possible_position)
                        next_queue.append(possible_position)
            queue = next_queue

    def minimum_moves(self, from_, to):
        visited = set()
        queue = [(from_, 0)]
        visited.add(from_)
        while queue:
            word, moves = queue.pop(0)
            for possible_position in self.possible_positions_in_1_move(word):
                if possible_position == to:
                    return moves + 1
                if possible_position not in visited:
                    visited.add(possible_position)
                    queue.append((possible_position, moves + 1))
        return None

    @lru_cache(maxsize=None)
    def minimum_path_v1(self, from_, to, exclude=None):
        if from_ == to:
            return [from_]
        previous = {}
        q = queue.Queue()
        q.put((from_, 0))
        previous[from_] = None
        while q:
            word, moves = q.get()
            for possible_position in self.possible_positions_in_1_move(word):
                if possible_position == to:
                    path = [possible_position, word]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])
                    return list(reversed(path))
                if possible_position not in previous and (exclude is None or possible_position != exclude):
                    previous[possible_position] = word
                    q.put((possible_position, moves + 1))
        return None

    def generate_first_step(self, from_, to):
        positions = defaultdict(list)
        for i, x in enumerate(from_):
            group = self.group_of_word(x)
            positions[group].append(i)

        matches = {x: 0 for x in positions}
        for x in to:
            group = self.group_of_word(x)
            if group == x:
                matches[group] += 1

        start_word = list(from_)
        generation_list = [(x, matches[x], positions[x]) for x in positions]

        def generate(generation_list, path, index=0):
            if index == len(generation_list):
                path = ["".join(x) for x in path]
                yield path
                return

            letter, match, positions = generation_list[index]
            permutted = [0] * match + [1] * (len(positions) - match)
            for permutation in more_itertools.distinct_permutations(permutted):
                current_path = list(path)
                word = list(path[-1])
                for i, x in enumerate(permutation):
                    position = positions[i]
                    final_letter = letter if x == 0 else self.transpositions[letter]
                    if word[position] == final_letter:
                        continue
                    word[position] = final_letter
                    current_path.append(word)
                    word = list(word)
                yield from generate(generation_list, current_path, index + 1)

        return generate(generation_list, [start_word])

    def longest_common_subsequence(self, from_: list[chr], to: list[chr]):
        n = len(from_)
        m = len(to)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if from_[i] == to[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        i = n
        j = m
        result = []
        while i > 0 and j > 0:
            if from_[i - 1] == to[j - 1]:
                result.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return list(reversed(result))

    @lru_cache(maxsize=None)
    def minimum_path(self, from_, to, exclude=None):
        if from_ == to:
            return [from_]
        min_path = None

        positions = defaultdict(list)
        for i, x in enumerate(to):
            positions[x].append(i)

        for path in self.generate_first_step(from_, to):
            from_ = path[-1]
            lcs = self.longest_common_subsequence(from_, to)
            # Transpose letter so that they match
            while len(lcs) != len(to):
                word = list(path[-1])
                fitted_positions = [x[1] for x in lcs]
                for i, (x, _) in enumerate(lcs):
                    if i == x:
                        continue
                    break
                else:
                    i += 1
                letter = word[i]
                for j in positions[letter]:
                    if j in fitted_positions:
                        continue
                    break
                for k in range(len(lcs)):
                    if lcs[k][1] > j:
                        break
                else:
                    k += 1
                k -= 1
                insert_after_index = lcs[k][0] if k >= 0 else -1
                if i <= insert_after_index:
                    word = word[:i] + word[i + 1 : insert_after_index + 1] + [letter] + word[insert_after_index + 1 :]
                    lcs = lcs[:i] + [(x - 1, y) for x, y in lcs[i : k + 1]] + [(insert_after_index, j)] + lcs[k + 1 :]
                else:
                    word = word[: insert_after_index + 1] + [letter] + word[insert_after_index + 1 : i] + word[i + 1 :]
                    lcs = lcs[: k + 1] + [(k + 1, j)] + [(x + 1, y) for x, y in lcs[k + 1 : i]] + lcs[i:]
                path.append("".join(word))
                another_lcs = self.longest_common_subsequence(word, to)
                assert len(lcs) == len(another_lcs)

            if min_path is None or len(min_path) > len(path):
                min_path = path
        if exclude is not None:
            if exclude in min_path:
                i = min_path.index(exclude)
                if i != -1:
                    res = self.minimum_path_v1(min_path[i - 1], min_path[i + 1], exclude=exclude)
                    assert len(res) == 3
                    min_path = min_path[:i] + [res[1]] + min_path[i + 1 :]
        return min_path

    def optimal_solution_one_seq(self, start, all_words):
        path = [start]
        for word in all_words:
            path = path[:-1] + self.minimum_path(path[-1], word)
        return path

    def optimal_solution(self, start, all_words, exclude=None):
        all_words = sorted(all_words)
        cache = {(tuple(), x): self.minimum_path(start, x, exclude=exclude) for x in tqdm(all_words, leave=False)}
        for subset_size in tqdm(range(1, len(all_words)), leave=False):
            for subset in itertools.combinations(all_words, subset_size):
                for word in all_words:
                    if word in subset:
                        continue
                    min_path = None
                    for connection_word in subset:
                        path = (
                            cache[tuple(x for x in subset if x != connection_word), connection_word]
                            + self.minimum_path(connection_word, word, exclude=exclude)[1:]
                        )
                        if min_path is None or len(path) < len(min_path):
                            min_path = path
                    cache[(subset, word)] = min_path

        min_path = None
        subset_size = len(all_words) - 1
        for subset in itertools.combinations(all_words, subset_size):
            for word in all_words:
                if word in subset:
                    continue
                path = cache[(subset, word)]
                if min_path is None or len(path) < len(min_path):
                    min_path = path
        return min_path

    def minimum_solution(self, start, all_words):
        min_path = None
        min_words = []
        for word in tqdm(all_words, leave=False):
            path = self.minimum_path(start, word)
            if min_path is None or len(path) < len(min_path):
                min_path = path
                min_words = [word]
            elif len(path) == len(min_path):
                min_path = path
                min_words.append(word)

        min_solution = None
        for word in tqdm(min_words, leave=False):
            words = list(all_words)
            words.remove(word)
            solution = self.optimal_solution(start, words, exclude=word)
            if word in solution:
                print(f"ERROR: {word} in solution {solution}")
                input()
            else:
                solution = solution + solution[-2::-1] + self.minimum_path(start, word)[1:]
                if min_solution is None or len(solution) < len(min_solution):
                    min_solution = solution

        return min_solution, len(min_path) - 1


s = Solver(words_filename="words.txt")
test(s.in_1_move, True, "daa", "daa")
test(s.in_1_move, True, "daa", "ada")
test(s.in_1_move, True, "daa", "aad")
test(s.in_1_move, True, "bps", "sbp")
test(s.in_1_move, True, "asdf", "sdfa")
test(s.in_1_move, False, "asdf", "dfas")
test(s.in_1_move, False, "fast", "fast")

test(s.in_1_move, True, "daa", "paa")
test(s.in_1_move, True, "daa", "dae")
test(s.in_1_move, False, "daa", "pae")
test(s.in_1_move, False, "daa", "apa")

seq = ["hue", "hen", "yen", "nay", "nah", "any"]
start = "adm"
all_words = ["dam", "paw", "pew", "wed", "mad", "amp", "map", "wad", "dew"]
# start = "ostp"
# all_words = ["stop", "pots", "spot", "post", "tops", "opts", "dots"]
# start = "ddo"
# all_words = ["odd", "pop", "pod"]

# start = "erar"
# all_words = ["rear", "rare", "jeer", "ajar"]
start = "aedp"
all_words = ["peep", "deep", "papa", "aped", "dead"]

levels = [
    {"letters": "em", "moves": [8, 6], "reqStars": 0},
    {"letters": "ddo", "moves": [8, 4], "reqStars": 1},
    {"letters": "eaw", "moves": [7, 3], "reqStars": 3},
    {"letters": "ueh", "moves": [15, 8], "reqStars": 5},
    {"letters": "adm", "moves": [16, 12, 10], "reqStars": 7},
    {"letters": "iskp", "moves": [14, 5], "reqStars": 10},
    {"letters": "ostp", "moves": [14, 7], "reqStars": 12},
    {"letters": "erar", "moves": [14, 7], "reqStars": 14},
    {"letters": "aedp", "moves": [13, 9], "reqStars": 16},
    {"letters": "aspn", "moves": [25, 18, 16], "reqStars": 18},
    {"letters": "dwas", "moves": [25, 15, 14], "reqStars": 20},
    {"letters": "oslva", "moves": [20, 10], "reqStars": 23},
    {"letters": "larat", "moves": [18, 6], "reqStars": 25},
    {"letters": "yseat", "moves": [18, 9], "reqStars": 27},
    {"letters": "urepe", "moves": [20, 10], "reqStars": 29},
    {"letters": "twsae", "moves": [22, 16, 15], "reqStars": 32},
    {"letters": "dsala", "moves": [30, 22, 21], "reqStars": 35},
    {"letters": "reedh", "moves": [28, 20, 18], "reqStars": 37},
    {"letters": "bdeeer", "moves": [10, 7], "reqStars": 40},
    {"letters": "ceortv", "moves": [16, 9], "reqStars": 41},
    {"letters": "ehortu", "moves": [20, 14], "reqStars": 43},
    {"letters": "epremp", "moves": [25, 19, 17], "reqStars": 45},
    {"letters": "eelstd", "moves": [38, 30, 27], "reqStars": 48},
    {"letters": "popejau", "moves": [30, 21], "reqStars": 51},
    {"letters": "csdieru", "moves": [30, 21], "reqStars": 53},
    {"letters": "dartaus", "moves": [42, 35, 32], "reqStars": 55},
    {"letters": "iouutesc", "moves": [27, 20, 19], "reqStars": 58},
    {"letters": "diugsije", "moves": [28, 20, 18], "reqStars": 61},
    {"letters": "ejesiduts", "moves": [38, 30, 27], "reqStars": 63},
    {"letters": "iaatricpus", "moves": [28, 20, 19], "reqStars": 66},
    {"letters": "caeretiultd", "moves": [30, 17, 16], "reqStars": 68},
]


def level_description(number, level):
    all_words = s.possible_words(level["letters"])
    optimal_solution = s.optimal_solution(level["letters"], all_words)
    star_symbol = "★"
    stars_description = "\n".join(
        f"{star_symbol * i}: {x} moves{'' if len(optimal_solution) <= x + 1 else ' (IMPOSSIBLE WITHOUT USING BUG)'}"
        for i, x in enumerate(level["moves"], start=1)
    )
    minimum_solution, minimum_moves = s.minimum_solution(level["letters"], all_words)
    description = f"""
Level {number} - {level["letters"]}
---------------------
Start word: {level["letters"]}
Stars to unlock: {level["reqStars"]}
{stars_description}
All words ([spoiler]{len(all_words)}[/spoiler] words): [spoiler]{", ".join(all_words)}[/spoiler]
Optimal solution ([spoiler]{len(optimal_solution) - 1}[/spoiler] moves): [spoiler]{" -> ".join(optimal_solution)}[/spoiler]
Minimum solution ([spoiler]{minimum_moves}[/spoiler] moves (bugged), actually [spoiler]{len(minimum_solution) - 1}[/spoiler]): [spoiler]{" -> ".join(minimum_solution)}[/spoiler]
    """.strip()
    return description


def generate_guide(levels):
    guide = []
    for i, level in enumerate(levels):
        guide.append(level_description(i, level))
        print(guide[-1], end="\n\n")
    return guide


guide = generate_guide(levels)
description = "\n\n".join(guide)
print(description)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file = sys.argv[1]
        with open(file, "w", encoding="utf-8") as f:
            f.write(description)
