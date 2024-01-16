import argparse
import builtins
import torch
from src.model import RNN
from src.time_logs import TimerLog
from src.data import build_librispeech

N_MELS = 256
MODEL_NAME = "model/RNN.pt"

def timed_print(*args):
    """ Print function to include elapsed time. """
    elapsed = TimerLog().get_elapsed()
    print_str = ' '.join(str(arg) for arg in args)
    print(f"{elapsed:.6f}\t| {print_str}")

def train_model(librispeech, args):
    """ Trains RNN with given dataset. """
    model = RNN(input_size=N_MELS, hidden_size=100, num_layers=1, device=args.device, verbose=args.verbose)
    for key in librispeech:
        if args.verbose:
            print(f"Dataset: {key}")
        model.train_model(librispeech[key], epochs=args.epochs, lrate=args.lrate)

    torch.save(model, MODEL_NAME)

def test_model(librispeech, args):
    """ Tests RNN with given dataset. """
    model = torch.load(MODEL_NAME)
    for key in librispeech:
        if args.verbose:
            print(f"Dataset: {key}")
        model.test_model(librispeech[key])

def main(args):
    """ Main program execution. """
    if args.timed:
        builtins.print = timed_print

    librispeech = build_librispeech(mode=args.mode, verbose=args.verbose)
    
    if args.mode == "training":
        if args.verbose:
            print("Beginning VAD model training...")
        train_model(librispeech, args)
        if args.verbose:
            print(f"Success! Finished training in {TimerLog().get_elapsed()} seconds.")
            print(f"Saving model to {MODEL_NAME}")

    elif args.mode == "testing":
        if args.verbose:
            print("Beginning VAD model testing...")
        test_model(librispeech, args)

    else:
        raise ValueError("Invalid mode selected. Please use CLI parameter '-m training' or '-m testing'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debugging logs")
    parser.add_argument("--timed", "-t", action="store_true", help="Display execution time in debugging logs")
    parser.add_argument("--mode", "-m", help="Set program to training or testing mode")
    parser.add_argument("--lrate", "-l", default=0.01, type=float, help="Set learning rate for model")
    parser.add_argument("--epochs", "-e", default=1, type=int, help="Set number of epochs for model training")
    parser.add_argument("--device", "-d", default='cuda' if torch.cuda.is_available() else 'cpu', help="Set program to run on CPU or GPU")
    args = parser.parse_args()

    main(args)
