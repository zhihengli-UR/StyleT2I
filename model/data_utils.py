def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
