# How to Contribute

We'd love to get patches from you!

## Workflow

We follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/)

1.  Fork the project
1.  Check out the `main` branch
1.  Create a feature branch
1.  Write code and tests for your change
1.  From your branch, make a pull request against `https://github.com/spotify-research/llark/main`
1.  Work with repo maintainers to get your change reviewed
1.  Wait for your change to be pulled into `https://github.com/spotify-research/llark/main`
1.  Delete your feature branch


## Style

We follow `black` style.

## Issues

When creating an issue please try to adhere to the following format:

    module-name: One line summary of the issue (less than 72 characters)

    ### Expected behavior

    As concisely as possible, describe the expected behavior.

    ### Actual behavior

    As concisely as possible, describe the observed behavior.

    ### Steps to reproduce the behavior

    List all relevant steps to reproduce the observed behavior.

## Pull Requests

Comments should be formatted to a width no greater than 100 columns.

Files should be exempt of trailing spaces.

We adhere to a specific format for commit messages. Please write your commit
messages along these guidelines. Please keep the line width no greater than 100
columns (You can use `fmt -n -p -w 100` to accomplish this).

    module-name: One line description of your change (less than 72 characters)

    Problem

    Explain the context and why you're making that change.  What is the problem
    you're trying to solve? In some cases there is not a problem and this can be
    thought of being the motivation for your change.

    Solution

    Describe the modifications you've done.

    Result

    What will change as a result of your pull request? Note that sometimes this
    section is unnecessary because it is self-explanatory based on the solution.

Some important notes regarding the summary line:

* Describe what was done; not the result
* Use the active voice
* Use the present tense
* Capitalize properly
* Do not end in a period — this is a title/subject
* Prefix the subject with its scope

## Code Review

The repository on GitHub is kept in sync with an internal repository at
Spotify. For the most part this process should be transparent to the project users,
but it does have some implications for how pull requests are merged into the
codebase.

When you submit a pull request on GitHub, it will be reviewed by the project
community (both inside and outside of Spotify), and once the changes are
approved, your commits will be brought into Spotify's internal system for
additional testing. Once the changes are merged internally, they will be pushed
back to GitHub with the next sync.

This process means that the pull request will not be merged in the usual way.
Instead a member of the project team will post a message in the pull request
thread when your changes have made their way back to GitHub, and the pull
request will be closed.
The changes in the pull request will be collapsed into a single commit, but the
authorship metadata will be preserved.

## Documentation

We also welcome improvements to the project documentation or to the existing
docs. Please file an [issue](https://github.com/spotify-research/llark/issues/new).

# License

By contributing your code, you agree to license your contribution under the
terms of the [LICENSE](https://github.com/spotify-research/llark/blob/main/LICENSE)

# Code of Conduct

Read our [Code of Conduct](CODE_OF_CONDUCT.md) for the project.
