from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='seq_simul',
      version='1.0',
      author="Sanghoon Kang, Yunfei Gao, and Albert No",
      author_email="albertno@hongik.ac.kr",
      description="DNA storage channel simulator",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/albert-no/seq-simul",
      packages=['seq_simul'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.7',
      )
