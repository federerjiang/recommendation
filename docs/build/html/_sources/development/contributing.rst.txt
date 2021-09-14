.. _contributing:

{{ header }}

**********************
Contributing to porise 
**********************

.. contents:: Table of contents:
   :local:

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.



Contributing to the code base
=====================

Now that you have an issue you want to fix, enhancement to add, or documentation to improve,
you need to learn how to work with Git and porise code base.


Creating a branch
-----------------

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to porise. You can have many shiny-new-features
and switch in between them using the git checkout command.

When creating this branch, make sure your master branch is up to date with
the latest upstream master version. To update your local master branch, you
can do::

    git checkout master
    git pull upstream master --ff-only

When you want to update the feature branch with changes in master after
you created the branch, check the section on
:ref:`updating a PR <contributing.update-pr>`.

Committing your code
--------------------

Keep style fixes to a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing::

    git status

If you have created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like::

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

Finally, commit your changes to your local repository with an explanatory message. porise
uses a convention for commit message prefixes and layout.  Here are
some common prefixes along with general guidelines for when to use them:

* ENH: Enhancement, new functionality
* BUG: Bug fix
* DOC: Additions/updates to documentation
* TST: Additions/updates to tests
* BLD: Updates to the build process/scripts
* PERF: Performance improvement
* TYP: Type annotations
* CLN: Code cleanup

The following defines how a commit message should be structured.  Please reference the
relevant Bitbucket issues in your commit message using GH1234 or #1234.  Either style
is fine, but the former is generally preferred:

* a subject line with ``< 80`` chars.
* One blank line.
* Optionally, a commit message body.

Now you can commit your changes in your local repository::

    git commit -m

.. _contributing.push-code:

Pushing your changes
--------------------

When you want your changes to appear publicly on your Bitbucket page, push your
forked feature branch's commits::

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on Bitbucket.
You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  git@Bitbucket.com:yourname/porise.git (fetch)
    origin  git@Bitbucket.com:yourname/porise.git (push)
    upstream        git://Bitbucket.com/porise-dev/porise.git (fetch)
    upstream        git://Bitbucket.com/porise-dev/porise.git (push)

Now your code is on Bitbucket, but it is not yet a part of the porise project. For that to
happen, a pull request needs to be submitted on Bitbucket.

Review your code
----------------

When you're ready to ask for a code review, file a pull request. Before you do, once
again make sure that you have followed all the guidelines outlined in this document
regarding code style, tests, performance tests, and documentation. You should also
double check your branch changes against the branch it was based on:

#. Navigate to your repository on Bitbucket -- https://Bitbucket.com/your-user-name/porise
#. Click on ``Branches``
#. Click on the ``Compare`` button for your feature branch
#. Select the ``base`` and ``compare`` branches, if necessary. This will be ``master`` and
   ``shiny-new-feature``, respectively.

Finally, make the pull request
------------------------------

If everything looks good, you are ready to make a pull request.  A pull request is how
code from a local repository becomes available to the Bitbucket community and can be looked
at and eventually merged into the master version.  This pull request and its associated
changes will eventually be committed to the master branch and available in the next
release.  To submit a pull request:

#. Navigate to your repository on Bitbucke
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code.

.. _contributing.update-pr:

Updating your pull request
--------------------------

Based on the review you get on your pull request, you will probably need to make
some changes to the code. In that case, you can make them in your branch,
add a new commit to that branch, push it to Bitbucket, and the pull request will be
automatically updated.  Pushing them to Bitbucket again is done by::

    git push origin shiny-new-feature

This will automatically update your pull request with the latest code and restart the
:ref:`Continuous Integration <contributing.ci>` tests.

Another reason you might need to update your pull request is to solve conflicts
with changes that have been merged into the master branch since you opened your
pull request.

To do this, you need to "merge upstream master" in your branch::

    git checkout shiny-new-feature
    git fetch upstream
    git merge upstream/master


If you have uncommitted changes at the moment you want to update the branch with
master, you will need to ``stash`` them prior to updating (see the
`stash docs <https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning>`__).
This will effectively store your changes and they can be reapplied after updating.

After the feature branch has been update locally, you can now update your pull
request by pushing to the branch on Bitbucket::

    git push origin shiny-new-feature


Tips for a successful pull request
==================================

If you have made it to the `Review your code`_ phase, one of the core contributors may
take a look. Please note however that a handful of people are responsible for reviewing
all of the contributions, which can often lead to bottlenecks.

To improve the chances of your pull request being reviewed, you should:

- **Reference an open issue** for non-trivial changes to clarify the PR's purpose
- **Ensure you have appropriate tests**. These should be the first part of any PR
- **Keep your pull requests as simple as possible**. Larger PRs take longer to review
- **Ensure that CI is in a green state**. Reviewers may not even look otherwise
- **Keep** `Updating your pull request`_, either by request or every few days

