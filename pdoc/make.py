#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""
# Note: Invoke this script from the parent directory as "pdoc/make.py".

import pathlib

import pdoc

OUTPUT_DIRECTORY = pathlib.Path('./docs')


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
      docformat='google',
      edit_url_map=None,
      # favicon='https://github.com/hhoppe/hhoppe-tools/raw/main/v.ico',
      footer_text='',
      # logo='https://github.com/hhoppe/hhoppe-tools/raw/main/v.png',
      logo_link='https://hhoppe.github.io/hhoppe-tools/',
      math=True,
      search=True,
      show_source=True,
      template_directory=pathlib.Path('./doc'),
  )

  pdoc.pdoc(
      './hhoppe_tools',
      output_directory=OUTPUT_DIRECTORY,
  )

  if 1:
    output_file = OUTPUT_DIRECTORY / 'hhoppe_tools.html'
    text = output_file.read_text()
    # collections.abc.Iterable -> Iterable.
    text = text.replace('<span class="n">collections</span><span class="o">'
                        '.</span><span class="n">abc</span><span class="o">.</span>', '')
    # typing.* -> *.
    text = text.replace('<span class="n">typing</span><span class="o">.</span>', '')
    output_file.write_text(text)


if __name__ == '__main__':
  main()
