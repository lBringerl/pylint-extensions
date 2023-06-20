import astroid


def main():
    with open('test.py') as f:
        module = astroid.parse(f.read())
    repr = module.repr_tree()

    gen = module.body[-1].value.infer()
    for v in gen:
        print(v.value)


if __name__ == '__main__':
    main()
