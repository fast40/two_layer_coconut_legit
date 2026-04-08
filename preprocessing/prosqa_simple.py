import json
import argparse
import re


def main(split):
    with open(f"data/prosqa_{split}.json") as f:
        data = json.load(f)

    simple_data = []

    for sample in data:
        question = sample["question"].replace("Every ", "")
        steps = " ".join(sample["steps"]).replace("Every ", "")
        symbols = {symbol: i for i, symbol in enumerate(sample["idx_to_symbol"])}
        edges_ground_truth = sample["edges"]
        target = sample["target"]
        neg_target = sample["neg_target"]
        root = sample["root"]

        get_edge_tuples = lambda edges_str: [
            [symbols[edge[0]], symbols[edge[1]]]
            for edge in re.findall(r"(\w+) is an? (\w+).", edges_str)
        ]

        edges = get_edge_tuples(question)
        steps = get_edge_tuples(steps)

        assert sorted(edges) == sorted(edges_ground_truth)

        final_question = re.search(r"Is (\w+) an? (\w+) or (\w+)?", question)

        assert final_question is not None  # make sure we got a question

        assert symbols[final_question.group(1)] == root  # make sure

        option1 = symbols[final_question.group(2)]
        option2 = symbols[final_question.group(3)]

        assert (option1 == target and option2 == neg_target) or (
            option1 == neg_target and option2 == target
        )

        q = (
            "<eos> "
            + " | ".join(f"{edge[0]} {edge[1]}" for edge in edges)
            + f" [Q] {option1} {option2} [R] {root}"
        )

        s = " | ".join(f"{step[0]} {step[1]}" for step in steps)

        simple_data.append({"question": q, "steps": s, "answer": str(target)})

    with open(f"data/prosqa_simple_{split}.json", "w") as f:
        json.dump(simple_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert standard prosqa json data to simple prosqa json data."
    )
    parser.add_argument(
        "split", type=str, help="The dataset split (e.g., train, test, valid)."
    )
    args = parser.parse_args()
    main(args.split)
