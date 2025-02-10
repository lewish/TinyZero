import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    solution_str = solution_str.split("<think>")[-1]

    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """The scoring function for rearc task.

    Args:
        solution_str: the solution text
        ground_truth: the expected solution text
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    answer = extract_solution(solution_str=solution_str)

    if answer:
        answer = answer.strip()
    if ground_truth:
        ground_truth = ground_truth.strip()

    do_print = random.randint(1, 16) == 1

    if do_print:
        print(f"DEBUG---------------------------")
        print(f"Full solution string:\n{solution_str}")
        print(f"Extracted solution:\n{answer}")
        print(f"Target:\n{ground_truth}")
        print(f"--------------------------------")

    if ground_truth == answer:
        if do_print:
            print(f"Correct answer!")
        return score

    if answer is not None:
        answer_lines = answer.split("\n")
        ground_truth_lines = ground_truth.split("\n")

        if len(answer_lines) == len(ground_truth_lines) and all(
            len(a.split()) == len(g.split())
            for a, g in zip(answer_lines, ground_truth_lines)
        ):
            if do_print:
                print(f"Correct grid size, wrong answer!")
            return format_score

        if re.match(r"^(\d\s)*\d$", answer):
            if do_print:
                print(f"Wrong answer!")
            # Try to reduce rewards for just putting out consistent values.
            return format_score / 10

    if do_print:
        print(f"Wrong format!")

    return 0


# No time to write real tests hrrr
if __name__ == "__main__":
    solution_str = """Assistant: Let me solve this step by step.
<answer>4</answer>
<think>
</think>
<answer>

1 2
3 4

</answer>"""
    ground_truth = "1 2\n3 4"
    score = compute_score(solution_str, ground_truth)
    print(f"Score: {score}")
