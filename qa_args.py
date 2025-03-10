import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str,
                        default='train-v1.1.json')
    parser.add_argument('--run-name', type=str, default='qa_bert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str,
                        default='data')
    parser.add_argument('--val-dir', type=str, default='data')
    parser.add_argument('--val-datasets', type=str,
                        default='dev-v1.1.json')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--pred-file', type=str, default='',
                        help="predictions from model")
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()

    return args