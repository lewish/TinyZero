import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if matches:
        if len(matches) == 1:
            # This is bad, as we put an example pair of answer tags in the prompt, so it means it didn't produce answer tags.
            return None
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

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"Full solution string:\n{solution_str}")
        print(f"Extracted solution:\n{answer}")
        print(f"Target:\n{ground_truth}")
        print(f"--------------------------------")

    if ground_truth == answer:
        if do_print:
            print(f"Correct answer: {answer}")
        return score

    if answer is not None and re.match(r"^(\d\s)*\d$", answer):
        if do_print:
            print(f"Wrong answer: {answer}")
        return format_score

    if do_print:
        print(f"Wrong format")

    return 0


# No time to write real tests hrrr
if __name__ == "__main__":
    solution_str = """Assistant: Let me solve this step by step.
<answer>4</answer>
<answer>

1 1
3 4

</answer>"""
    ground_truth = "1 2\n3 4"
    score = compute_score(solution_str, ground_truth)
    print(f"Score: {score}")
