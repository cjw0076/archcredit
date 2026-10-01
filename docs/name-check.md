# Package name check

Checked on 2026-10-01 before repository creation. The selected neutral name is **archcredit**, combining architecture and credit assignment without making Cosmos the entire package identity.

| Candidate | PyPI JSON endpoint | GitHub `in:name` search count | Decision |
| --- | --- | --- | --- |
| `archcredit` | HTTP 404 | 0 | Selected |
| `neurocredit` | HTTP 404 | 5 | Existing financial-project associations |
| `creditlab` | HTTP 404 | 11 | Existing financial-project associations |

The PyPI checks used `/pypi/<candidate>/json` on `pypi.org`; a 404 indicates no project at that endpoint at the time. GitHub checks used the repository search API with `<candidate> in:name`; that search can include partial name matches and is not an exact identifier reservation.

The five `neurocredit` search results were `mallochio/neuroCredit`, `carlos-alves-one/NeuroCredit`, `Donwload/neuroCredit`, `EnakxD/NeuroCredit`, and `Trust-Anchor-Group/NeuroCredits`.

The public repository was created at [cjw0076/archcredit](https://github.com/cjw0076/archcredit). PyPI publication is a separate release step. These checks are neither trademark clearance nor a package-name reservation; repeat them immediately before release.
