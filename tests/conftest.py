import torch


def pytest_sessionstart(session):
    # Small numerical tests are much faster without oversubscribing CPU threads.
    torch.set_num_threads(2)
