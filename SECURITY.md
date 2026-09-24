# Security

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on this repository. Do not open a public
issue for anything that could put users' tokens or data at risk.

The CLI stores your access token in its config directory with `0600`
permissions and sends it only to the registry endpoint you configured. You can
revoke any token at <https://systemonemodels.tech/settings/tokens>.
