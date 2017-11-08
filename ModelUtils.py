
def create_model_class(name):
    module_name = "model_" + name
    module = __import__(module_name)
    class_name = "Model_" + name
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def parse_epoch(snapshot):
    i = snapshot.find('-')
    if i == -1:
        return 0
    j = snapshot.find('-', i+1)
    if j == -1:
        return 0
    return int(snapshot[i+1:j])
