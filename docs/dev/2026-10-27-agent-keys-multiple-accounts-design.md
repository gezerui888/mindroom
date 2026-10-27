# Several Accounts per Provider for Agent Keys

This record explains how an agent's shell and python can use more than one account of the same OAuth provider through Agent Keys, and why the extra accounts are separate from the connection the agent's tools use.

## Why

Agent Keys hands worker requests the scope's one connection per provider, the same one the agent's tools use.
Agent Vault, which it replaced, let a vault hold several logins, such as a user's own Google account and a second Google account for a shared mailbox or another calendar.
Nothing in Agent Keys can express that, so a migration had to drop those logins or map them onto the tool connection, which belongs to a different account.

## Decisions

- **An extra account is a labelled connection of an existing provider.** A label matches `^[a-z0-9][a-z0-9-]{0,30}$`; the account is stored in the same scope store as the provider's own connection, under the credential service `f"{base}__{label}_oauth"`, where `base` is the provider's credential service without its `_oauth` suffix (`google_drive__work_oauth`).
  The label goes before the suffix because provider validation and credential placement treat only `*_oauth` names as OAuth tokens; any other name would fail validation, and placement would treat it as a worker-grantable setting.
  Every OAuth mechanism already works per credential service (the connect flow, the callback, refresh, the credential lifecycle), so a labelled account is the provider with that one field replaced (`dataclasses.replace`), and needs no new storage.
  The provider's own connection keeps its name and stays the one the tools use; tools never see labelled accounts.
- **Connecting an extra account goes through Agent Keys.** The Agent Keys connect endpoint takes an optional `account` label, the OAuth pending state records it, and the callback stores the result under the labelled credential service.
  The disconnect endpoint takes the same label and removes only that account.
  Who may connect or disconnect follows the existing rules for the scope (the requester for a personal agent; credential managers and administrators for a shared one).
- **Each account is allowed per service, like the provider's own connection.** The opt-in document keeps one entry per allowed (service, account) pair: the service name for the provider's own connection, as today, and `f"{service}@{label}"` for a labelled account.
  Existing documents stay valid unchanged, and turning one account off leaves the others.
  An entry carries the same fields as today's (`provider`, `enabled_by`, `enabled_at`, and `access` once read-only access lands).
- **The worker picks the account per request with a header.** Routing still picks the service by host and path; a request may add `X-Agent-Keys-Account: <label>` to use that labelled account of the service's provider.
  The broker always removes the header before forwarding, so it never reaches the API.
  Without the header the request uses the provider's own connection, as today; a stored API key for the service still wins only then.
  A label that is not connected, or not allowed for the service, is refused with 403 and refusal code `account_not_allowed`, which names the label and the Agent Keys page, without reading any connection.
- **The agent learns which accounts exist.** `list_egress_credentials` lists, per service, its allowed accounts with label and account email, so the agent knows which header value to send; the agent-facing docs show the header.
- **The dashboard manages them on the service row.** "Add another account" asks for a label and starts the connect flow; each connected account shows its label and email, its allow toggle, and Disconnect.

## Later, not now

- **Other scopes than the provider's.** A labelled account gets the provider's scopes; permissions the provider does not request (for example a Google admin API) still need a provider of their own.
- **Choosing an account by path instead of a header**, for tools that cannot set headers.

## Assumptions

- Every place that reads or writes a connection resolves its store through the provider's `credential_service`, so a provider copy with a labelled `credential_service` reaches only that account's store.
- Squid hands the broker every credentialed request with its headers intact.

## Testing

- Label validation, and the labelled credential service name.
- Connect and callback: an `account` label is carried through the pending state and stored under the labelled name; the provider's own connection is untouched; disconnect with a label removes only that account.
- Opt-in: entries for the provider's own connection and for labelled accounts coexist and toggle independently; existing documents read unchanged.
- Broker: no header uses the provider's own connection; a header picks the allowed labelled account's token; an unknown or not-allowed label is refused with `account_not_allowed` without reading a connection; the header is never forwarded.
- Tool and dashboard: accounts and labels appear in `list_egress_credentials` and on the service row, and adding an account starts the connect flow with the label.
