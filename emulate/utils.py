

def markdown_class_method(cls, method):
    import inspect

    method_lines = inspect.getsource(getattr(cls, method))
    return (
        "```python\n"
        f"class {cls.__name__}:\n"
        "...\n"
        f"{method_lines}...\n"
        "```"
    )


def jupyter_show_class_method(cls, method):
    from IPython.display import display, Markdown
    return display(Markdown(markdown_class_method(cls, method)))
