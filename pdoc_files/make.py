#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""

# Note: Invoke this from the parent directory as "python3 pdoc_files/make.py".

import pathlib

import pdoc

OUTPUT_DIRECTORY = pathlib.Path('pdoc_files/html')


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
      docformat='google',
      edit_url_map=None,
      # favicon='https://github.com/hhoppe/hhoppe-tools/raw/main/v.ico',
      footer_text='',
      # logo='https://github.com/hhoppe/hhoppe-tools/raw/main/v2.png',
      logo_link='https://hhoppe.github.io/hhoppe-tools/',
      math=True,
      search=True,
      show_source=True,
      # template_directory=pathlib.Path('/pdoc_files'),
  )

  pdoc.pdoc(
      './hhoppe_tools',
      output_directory=OUTPUT_DIRECTORY,
  )

  if 1:
    output_file = OUTPUT_DIRECTORY / 'hhoppe_tools.html'
    text = output_file.read_text(encoding='utf-8')
    # typing.Any -> Any, in the signatures.  (Other rewrites, e.g. of collections.abc.* and
    # typing.*, would only alter the displayed source code.)
    text = text.replace(
        '<span class="n">typing</span><span class="o">.</span><span class="n">Any<',
        '<span class="n">Any<',
    )
    output_file.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
  main()
