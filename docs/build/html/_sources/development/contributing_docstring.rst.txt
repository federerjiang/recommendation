.. _docstring:

{{ header }}

======================
porise docstring guide
======================

About docstrings and standards
------------------------------

A Python docstring is a string used to document a Python module, class,
function or method, so programmers can understand what it does without having
to read the details of the implementation.

Also, it is a common practice to generate online (html) documentation
automatically from docstrings. `Sphinx <https://www.sphinx-doc.org>`_ serves
this purpose.

The next example gives an idea of what a docstring looks like:

.. code-block:: python

    def add(num1, num2):
        """
        Add up two integer numbers.

        This function simply wraps the ``+`` operator, and does not
        do anything interesting, except for illustrating what
        the docstring of a very simple function looks like.

        Parameters
        ----------
        num1 : int
            First number to add.
        num2 : int
            Second number to add.

        Returns
        -------
        int
            The sum of ``num1`` and ``num2``.

        See Also
        --------
        subtract : Subtract one integer from another.

        Examples
        --------
        >>> add(2, 2)
        4
        >>> add(25, 0)
        25
        >>> add(10, -10)
        0
        """
        return num1 + num2

In the case of porise, the NumPy docstring convention is followed. These conventions are
explained in this document:

* `numpydoc docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
  (which is based in the original `Guide to NumPy/SciPy documentation
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_)

numpydoc is a Sphinx extension to support the NumPy docstring convention.

The standard uses reStructuredText (reST). reStructuredText is a markup
language that allows encoding styles in plain text files. Documentation
about reStructuredText can be found in:

* `Sphinx reStructuredText primer <https://www.sphinx-doc.org/en/stable/rest.html>`_
* `Quick reStructuredText reference <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_
* `Full reStructuredText specification <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html>`_


The rest of this document will summarize all the above guidelines.


.. _docstring.tutorial:

Writing a docstring
-------------------

.. _docstring.general:

General rules
~~~~~~~~~~~~~

Docstrings must be defined with three double-quotes. No blank lines should be
left before or after the docstring. The text starts in the next line after the
opening quotes. The closing quotes have their own line
(meaning that they are not at the end of the last sentence).


.. _docstring.short_summary:

Section 1: short summary
~~~~~~~~~~~~~~~~~~~~~~~~

The short summary is a single sentence that expresses what the function does in
a concise way.

The short summary must start with a capital letter, end with a dot, and fit in
a single line. It needs to express what the object does without providing
details. For functions and methods, the short summary must start with an
infinitive verb.

**Good:**

.. code-block:: python

    def astype(dtype):
        """
        Cast Series type.

        This section will provide further details.
        """
        pass

**Bad:**

.. code-block:: python

    def astype(dtype):
        """
        Casts Series type.

        Verb in third-person of the present simple, should be infinitive.
        """
        pass

.. code-block:: python

    def astype(dtype):
        """
        Method to cast Series type.

        Does not start with verb.
        """
        pass

.. code-block:: python

    def astype(dtype):
        """
        Cast Series type

        Missing dot at the end.
        """
        pass

.. code-block:: python

    def astype(dtype):
        """
        Cast Series type from its current type to the new type defined in
        the parameter dtype.

        Summary is too verbose and doesn't fit in a single line.
        """
        pass

.. _docstring.extended_summary:

Section 2: extended summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extended summary provides details on what the function does. It should not
go into the details of the parameters, or discuss implementation notes, which
go in other sections.

A blank line is left between the short summary and the extended summary.
Every paragraph in the extended summary ends with a dot.

The extended summary should provide details on why the function is useful and
their use cases, if it is not too generic.

.. code-block:: python

    def unstack():
        """
        Pivot a row index to columns.

        When using a MultiIndex, a level can be pivoted so each value in
        the index becomes a column. This is especially useful when a subindex
        is repeated for the main index, and data is easier to visualize as a
        pivot table.

        The index level will be automatically removed from the index when added
        as columns.
        """
        pass

.. _docstring.parameters:

Section 3: parameters
~~~~~~~~~~~~~~~~~~~~~

The details of the parameters will be added in this section. This section has
the title "Parameters", followed by a line with a hyphen under each letter of
the word "Parameters". A blank line is left before the section title, but not
after, and not between the line with the word "Parameters" and the one with
the hyphens.

After the title, each parameter in the signature must be documented, including
``*args`` and ``**kwargs``, but not ``self``.

The parameters are defined by their name, followed by a space, a colon, another
space, and the type (or types). Note that the space between the name and the
colon is important. Types are not defined for ``*args`` and ``**kwargs``, but must
be defined for all other parameters. After the parameter definition, it is
required to have a line with the parameter description, which is indented, and
can have multiple lines. The description must start with a capital letter, and
finish with a dot.

For keyword arguments with a default value, the default will be listed after a
comma at the end of the type. The exact form of the type in this case will be
"int, default 0". In some cases it may be useful to explain what the default
argument means, which can be added after a comma "int, default -1, meaning all
cpus".

In cases where the default value is ``None``, meaning that the value will not be
used. Instead of ``"str, default None"``, it is preferred to write ``"str, optional"``.
When ``None`` is a value being used, we will keep the form "str, default None".
For example, in ``df.to_csv(compression=None)``, ``None`` is not a value being used,
but means that compression is optional, and no compression is being used if not
provided. In this case we will use ``"str, optional"``. Only in cases like
``func(value=None)`` and ``None`` is being used in the same way as ``0`` or ``foo``
would be used, then we will specify "str, int or None, default None".

**Good:**

.. code-block:: python

    class Series:
        def plot(self, kind, color='blue', **kwargs):
            """
            Generate a plot.

            Render the data in the Series as a matplotlib plot of the
            specified kind.

            Parameters
            ----------
            kind : str
                Kind of matplotlib plot.
            color : str, default 'blue'
                Color name or rgb code.
            **kwargs
                These parameters will be passed to the matplotlib plotting
                function.
            """
            pass

**Bad:**

.. code-block:: python

    class Series:
        def plot(self, kind, **kwargs):
            """
            Generate a plot.

            Render the data in the Series as a matplotlib plot of the
            specified kind.

            Note the blank line between the parameters title and the first
            parameter. Also, note that after the name of the parameter ``kind``
            and before the colon, a space is missing.

            Also, note that the parameter descriptions do not start with a
            capital letter, and do not finish with a dot.

            Finally, the ``**kwargs`` parameter is missing.

            Parameters
            ----------

            kind: str
                kind of matplotlib plot
            """
            pass

.. _docstring.parameter_types:

Parameter types
^^^^^^^^^^^^^^^

When specifying the parameter types, Python built-in data types can be used
directly (the Python type is preferred to the more verbose string, integer,
boolean, etc):

* int
* float
* str
* bool

For complex types, define the subtypes. For ``dict`` and ``tuple``, as more than
one type is present, we use the brackets to help read the type (curly brackets
for ``dict`` and normal brackets for ``tuple``):

* list of int
* dict of {str : int}
* tuple of (str, int, int)
* tuple of (str,)
* set of str

In case where there are just a set of values allowed, list them in curly
brackets and separated by commas (followed by a space). If the values are
ordinal and they have an order, list them in this order. Otherwise, list
the default value first, if there is one:

* {0, 10, 25}
* {'simple', 'advanced'}
* {'low', 'medium', 'high'}
* {'cat', 'dog', 'bird'}

If the type is defined in a Python module, the module must be specified:

* datetime.date
* datetime.datetime
* decimal.Decimal

If the type is in a package, the module must be also specified:

* numpy.ndarray
* scipy.sparse.coo_matrix


If the exact type is not relevant, but must be compatible with a NumPy
array, array-like can be specified. If Any type that can be iterated is
accepted, iterable can be used:

* array-like
* iterable

If more than one type is accepted, separate them by commas, except the
last two types, that need to be separated by the word 'or':

* int or float
* float, decimal.Decimal or None
* str or list of str

If ``None`` is one of the accepted values, it always needs to be the last in
the list.

For axis, the convention is to use something like:

* axis : {0 or 'index', 1 or 'columns', None}, default None

.. _docstring.returns:

Section 4: returns or yields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the method returns a value, it will be documented in this section. Also
if the method yields its output.

The title of the section will be defined in the same way as the "Parameters".
With the names "Returns" or "Yields" followed by a line with as many hyphens
as the letters in the preceding word.

The documentation of the return is also similar to the parameters. But in this
case, no name will be provided, unless the method returns or yields more than
one value (a tuple of values).

The types for "Returns" and "Yields" are the same as the ones for the
"Parameters". Also, the description must finish with a dot.

For example, with a single value:

.. code-block:: python

    def sample():
        """
        Generate and return a random number.

        The value is sampled from a continuous uniform distribution between
        0 and 1.

        Returns
        -------
        float
            Random number generated.
        """
        return np.random.random()

With more than one value:

.. code-block:: python

    import string

    def random_letters():
        """
        Generate and return a sequence of random letters.

        The length of the returned string is also random, and is also
        returned.

        Returns
        -------
        length : int
            Length of the returned string.
        letters : str
            String of random letters.
        """
        length = np.random.randint(1, 10)
        letters = ''.join(np.random.choice(string.ascii_lowercase)
                          for i in range(length))
        return length, letters

If the method yields its value:

.. code-block:: python

    def sample_values():
        """
        Generate an infinite sequence of random numbers.

        The values are sampled from a continuous uniform distribution between
        0 and 1.

        Yields
        ------
        float
            Random number generated.
        """
        while True:
            yield np.random.random()

