import { type FormEvent, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  accountReuseDescription,
  egressDisconnectDescription,
} from "./accountCopy";
import { ConfirmDialog } from "./ConfirmDialog";
import {
  canEdit,
  limitedNote,
  limitOverlap,
  limitOf,
  limitTarget,
  removeWarning,
  type RowAgent,
  type ServiceEditing,
  servicesUsingAccount,
  sharesAccount,
  unsetNote,
} from "./egressRowText";
import { BusyNote, useLimitActions } from "./EgressLimitRows";
import { ServiceStatus } from "./EgressServiceStatus";
import {
  AgentSees,
  type RowAction,
  RowActionsMenu,
  SERVICE_CELL,
  SERVICE_ROW,
  ServiceHosts,
} from "./EgressServiceTable";
import { coversGithub } from "./githubLimits";
import { connectWithPopup, type OAuthAuthorization } from "./oauthPopup";
import {
  errorText,
  type RequestErrorMessages,
  requestConnection,
} from "./request";
import { SharedAccountDialog } from "./SharedAccountDialog";
import { ItemName } from "./ToolRow";
import type { EgressCredentialService, EgressLabelledAccount } from "./types";

/** The endpoints one service row talks to. */
export interface ServicePaths {
  secret: string;
  connect: string;
  disconnect: string;
  /** Where Agent Keys' use of the account is allowed (PUT) or stopped (DELETE). */
  reuse: string;
  /**
   * Whether the service's account, or its labelled `account`, is connected
   * now, read again from the listing the row came from.
   */
  accountConnected: (signal: AbortSignal, account?: string) => Promise<boolean>;
}

// What the server needs before Agent Keys may use an account: the user saw that commands skip the tools' approvals.
const REUSE_BODY = { acknowledge_approval_bypass: true };
// The server's account label rule.
const ACCOUNT_LABEL = /^[a-z0-9][a-z0-9-]{0,30}$/;
const ACCOUNT_LABEL_HINT =
  "Use 1 to 31 lowercase letters, digits, or hyphens, starting with a letter or digit.";

/** Where Agent Keys' use of a labelled account is stopped: the reuse path with the label as `account`. */
const withAccount = (path: string, account: string) =>
  `${path}${path.includes("?") ? "&" : "?"}account=${encodeURIComponent(account)}`;

/** Asked before Agent Keys may use an account; a second click while it closes goes on only once. */
function AccountReuseDialog({
  open,
  provider,
  description,
  confirmLabel,
  onContinue,
  onCancel,
}: {
  open: boolean;
  provider: string;
  description: string;
  confirmLabel: string;
  onContinue: () => void;
  onCancel: () => void;
}) {
  const continued = useRef(false);
  useEffect(() => {
    if (open) continued.current = false;
  }, [open]);
  return (
    <ConfirmDialog
      open={open}
      title={`Use your ${provider} account for Agent Keys?`}
      description={description}
      confirmLabel={confirmLabel}
      destructive={false}
      onCancel={onCancel}
      onConfirm={() => {
        if (continued.current) return;
        continued.current = true;
        onContinue();
      }}
    />
  );
}

export function EgressServiceItem({
  paths,
  service,
  onChanged,
  errorMessages,
  serviceEditing,
  agent,
  hideKeyActions,
}: {
  paths: ServicePaths;
  service: EgressCredentialService;
  onChanged: () => void;
  errorMessages?: RequestErrorMessages;
  serviceEditing: ServiceEditing;
  agent: RowAgent;
  hideKeyActions: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [secret, setSecret] = useState("");
  const [busy, setBusy] = useState<
    "save" | "remove" | "connect" | "disconnect" | "reuse" | "delete" | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<
    "key" | "account" | "service" | "shared" | "reuse" | null
  >(null);
  // The label typed for another account of the provider, while its form is open.
  const [newAccount, setNewAccount] = useState<string | null>(null);
  // The labelled account a dialog asks about, and whether it allows or disconnects it.
  const [accountConfirm, setAccountConfirm] = useState<{
    action: "reuse" | "disconnect";
    account: EgressLabelledAccount;
  } | null>(null);
  const operation = useRef<AbortController | null>(null);
  const keyLabel = `${service.display_name} API key`;
  const oauth = service.oauth;
  // Rows with an account keep the key behind a toggle until a key is set.
  const showKeyControls = !oauth || service.key_configured || editing;
  const canManageKey = service.can_manage && !hideKeyActions;
  const keyInsteadLabel = oauth?.connected
    ? "Use an API key instead of the account"
    : "Use an API key instead";
  // Requester-scoped accounts belong to the requester, so the server lets any
  // eligible user manage them even without key rights.
  const canResetAccount =
    oauth !== null &&
    !hideKeyActions &&
    (oauth.connected || oauth.reset_required) &&
    (service.can_manage || oauth.can_connect);
  // Allowing or stopping Agent Keys' use of the account follows the same rule,
  // where the broker can use the account at all.
  const canChangeReuse =
    oauth !== null &&
    !hideKeyActions &&
    oauth.unavailable_reason === null &&
    (service.can_manage || oauth.can_connect);
  const reuseEnabled = oauth?.reuse_enabled === true;
  // Only the personal listing sends labelled accounts, and only Agent Keys connects them.
  const accounts = service.accounts;
  const canAddAccount =
    accounts !== undefined &&
    canChangeReuse &&
    oauth.can_connect &&
    !oauth.reset_required;

  // A limit of the viewer's that narrows this administrator's service.
  const limit = limitOf(service, agent.services);
  // Another service of the viewer's with rules on the limit's hosts.
  const overlap =
    limit &&
    limitOverlap(
      limit.rules.map((rule) => rule.host),
      agent.services,
    );
  // A service that already has a limit changes or removes it instead of getting another.
  const canLimit =
    limit === null &&
    serviceEditing.onLimitToRepositories !== undefined &&
    service.source === "config" &&
    service.can_manage &&
    coversGithub(service.rules);
  // Everyone using a shared agent acts as the account connected there.
  const asksAboutSharedAccount = sharesAccount(agent, oauth);
  const limitedText = limitedNote(service);
  const unsetText = unsetNote(service);
  const limitActions = useLimitActions(
    limit,
    service.display_name,
    serviceEditing,
    setError,
  );

  useEffect(() => () => operation.current?.abort(), []);

  const openKeyForm = () => {
    setError(null);
    setEditing(true);
  };
  const closeEditor = () => {
    setEditing(false);
    setSecret("");
  };

  /**
   * Runs one key or account action: the row is busy and shows its error, a
   * saved key closes the key form, and the page reloads the row after it. An
   * action abandoned by leaving the page changes nothing.
   */
  const run = async (
    kind: "save" | "remove" | "connect" | "disconnect" | "reuse",
    failure: string,
    action: (signal: AbortSignal) => Promise<unknown>,
  ) => {
    setConfirm(null);
    setAccountConfirm(null);
    const controller = new AbortController();
    operation.current = controller;
    setBusy(kind);
    setError(null);
    try {
      await action(controller.signal);
      if (controller.signal.aborted) return;
      if (kind === "save") closeEditor();
      onChanged();
    } catch (cause) {
      if (!controller.signal.aborted) setError(errorText(cause, failure));
    } finally {
      if (!controller.signal.aborted) setBusy(null);
    }
  };
  const write =
    (path: string, method: string, body?: object) => (signal: AbortSignal) =>
      requestConnection<void>(path, signal, method, body, errorMessages);

  const save = (event: FormEvent) => {
    event.preventDefault();
    if (!secret.trim() || busy) return;
    void run(
      "save",
      "Could not save the key. Try again.",
      write(paths.secret, "PUT", { secret }),
    );
  };

  const deleteService = async () => {
    setConfirm(null);
    setBusy("delete");
    setError(null);
    try {
      await serviceEditing.onDelete(service);
    } catch (cause) {
      setError(errorText(cause, "Could not delete the service. Try again."));
    } finally {
      setBusy(null);
    }
  };

  const connect = () => {
    // A second click on Continue, while the first connect is under way, starts nothing.
    if (!oauth || busy !== null) return;
    void run("connect", "Could not connect. Try again.", (signal) =>
      connectWithPopup(
        oauth.provider,
        async () => {
          // Connecting here is for Agent Keys, so it allows the account first; the popup is already open.
          if (!oauth.reuse_enabled)
            await requestConnection<void>(
              paths.reuse,
              signal,
              "PUT",
              REUSE_BODY,
              errorMessages,
            );
          return requestConnection<OAuthAuthorization>(
            paths.connect,
            signal,
            "POST",
            {},
            errorMessages,
          );
        },
        signal,
        // A popup that closed after connecting is a connection, not a cancel.
        () => paths.accountConnected(signal),
      ),
    );
  };

  // Why the typed label cannot be connected, if it cannot.
  const newAccountProblem =
    newAccount === null || newAccount === ""
      ? null
      : !ACCOUNT_LABEL.test(newAccount)
        ? ACCOUNT_LABEL_HINT
        : accounts?.some((account) => account.label === newAccount)
          ? `${newAccount} already has an account. Reconnect or disconnect it instead.`
          : null;
  const connectAccount = (account: string) => {
    if (!oauth || busy !== null) return;
    void run("connect", "Could not connect. Try again.", async (signal) => {
      await connectWithPopup(
        oauth.provider,
        () =>
          requestConnection<OAuthAuthorization>(
            paths.connect,
            signal,
            "POST",
            { account },
            errorMessages,
          ),
        signal,
        () => paths.accountConnected(signal, account),
      );
      setNewAccount(null);
    });
  };

  const name = service.display_name;
  // Connecting reads the saved connection first, so an unreadable one needs a reset before.
  const canConnect =
    oauth !== null &&
    !hideKeyActions &&
    oauth.can_connect &&
    !oauth.connected &&
    !oauth.reset_required;
  // A connected account only its tools use, which Agent Keys may use once allowed; a key wins over it anyway.
  const canUseAccount =
    canChangeReuse &&
    oauth.connected &&
    !reuseEnabled &&
    !oauth.reset_required &&
    !service.key_configured;
  const keyActions = canManageKey && service.key_configured && !editing;
  // A key is the main action when there is no account to connect instead.
  const setKeyFirst =
    canManageKey &&
    !service.key_configured &&
    !editing &&
    (!oauth || (!canConnect && !oauth.connected && !oauth.reset_required));
  const editable = canEdit(serviceEditing, service);
  // Everything but the action that makes the service work, which is a button of its own.
  const actions: RowAction[] = [
    ...(oauth && canManageKey && !showKeyControls && !setKeyFirst
      ? [
          {
            label: keyInsteadLabel,
            ariaLabel: `${keyInsteadLabel} for ${name}`,
            onSelect: openKeyForm,
          },
        ]
      : []),
    ...(keyActions
      ? [
          {
            label: "Replace key",
            ariaLabel: `Replace key for ${name}`,
            onSelect: openKeyForm,
          },
        ]
      : []),
    ...(oauth && canAddAccount
      ? [
          {
            label: "Add another account",
            ariaLabel: `Add another ${oauth.display_name} account for ${name}`,
            onSelect: () => {
              setError(null);
              setNewAccount("");
            },
          },
        ]
      : []),
    ...(canLimit && serviceEditing.onLimitToRepositories
      ? [
          {
            label: "Limit to repositories",
            ariaLabel: `Limit to repositories for ${name}`,
            disabled: serviceEditing.disabled || serviceEditing.limitDisabled,
            onSelect: () => serviceEditing.onLimitToRepositories?.(service),
          },
        ]
      : []),
    ...limitActions.actions,
    ...(editable
      ? [
          {
            label: "Edit service",
            ariaLabel: `Edit service for ${name}`,
            disabled:
              serviceEditing.disabled ||
              serviceEditing.editingName === service.name,
            onSelect: () => serviceEditing.onEdit(service),
          },
        ]
      : []),
    ...(keyActions
      ? [
          {
            label: "Remove key",
            ariaLabel: `Remove key for ${name}`,
            destructive: true,
            onSelect: () => setConfirm("key"),
          },
        ]
      : []),
    ...(oauth && reuseEnabled && canChangeReuse
      ? [
          {
            label: `Stop using ${oauth.display_name} account`,
            ariaLabel: `Stop using ${oauth.display_name} account for ${name}`,
            onSelect: () =>
              void run(
                "reuse",
                "Could not stop using the account. Try again.",
                write(paths.reuse, "DELETE"),
              ),
          },
        ]
      : []),
    ...(oauth?.connected && canResetAccount
      ? [
          {
            label: "Disconnect",
            ariaLabel: `Disconnect for ${name}`,
            destructive: true,
            onSelect: () => setConfirm("account"),
          },
        ]
      : []),
    ...(editable
      ? [
          {
            label: "Delete service",
            ariaLabel: `Delete service for ${name}`,
            disabled: serviceEditing.disabled,
            destructive: true,
            onSelect: () => setConfirm("service"),
          },
        ]
      : []),
  ];
  const busyNote =
    busy === "remove" || limitActions.busy
      ? "Removing…"
      : busy === "disconnect"
        ? "Disconnecting…"
        : busy === "delete"
          ? "Deleting…"
          : busy === "reuse"
            ? "Saving…"
            : null;

  return (
    <tr aria-label={service.display_name} className={SERVICE_ROW}>
      {/* What the service is, what the agent sees, and the limit that narrows it. */}
      <th scope="row" className={SERVICE_CELL.name}>
        <ItemName
          names={[
            service.name,
            service.display_name,
            ...(oauth ? [oauth.provider] : []),
          ]}
          title={service.display_name}
          description={service.description}
        >
          <AgentSees service={service} />
          {limitedText && (
            <p className="text-xs text-muted-foreground">{limitedText}</p>
          )}
          {limit && (
            <p className="text-xs text-muted-foreground">
              {overlap
                ? `Limited to ${limitTarget(limit)}. ${overlap}`
                : `Limited to ${limitTarget(limit)}`}
            </p>
          )}
        </ItemName>
      </th>
      <td data-label="Hosts" className={SERVICE_CELL.hosts}>
        <ServiceHosts service={service} />
      </td>
      <td data-label="Status" className={SERVICE_CELL.status}>
        <div className="flex flex-col items-start gap-1">
          <ServiceStatus
            service={service}
            agent={agent}
            serviceEditing={serviceEditing}
          />
          {unsetText && (
            <span className="text-xs text-muted-foreground">{unsetText}</span>
          )}
          {oauth && accounts && accounts.length > 0 && (
            <ul
              aria-label={`Other ${oauth.display_name} accounts`}
              className="flex flex-col gap-1 text-xs"
            >
              {accounts.map((account) => (
                <li
                  key={account.label}
                  className="flex flex-wrap items-center gap-2"
                >
                  <span className="font-medium">{account.label}</span>
                  {account.connected ? (
                    account.account_label && (
                      <span className="text-muted-foreground">
                        {account.account_label}
                      </span>
                    )
                  ) : (
                    <span className="text-muted-foreground">Not connected</span>
                  )}
                  {!account.connected && canAddAccount && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 px-2 text-xs"
                      disabled={busy !== null}
                      aria-label={`Reconnect ${account.label} for ${name}`}
                      onClick={() => connectAccount(account.label)}
                    >
                      Reconnect
                    </Button>
                  )}
                  {canChangeReuse && (
                    <>
                      <label className="flex items-center gap-1">
                        <Checkbox
                          checked={account.reuse_enabled}
                          disabled={busy !== null}
                          aria-label={`Agent Keys may use ${account.label} for ${name}`}
                          onCheckedChange={() =>
                            account.reuse_enabled
                              ? void run(
                                  "reuse",
                                  "Could not stop using the account. Try again.",
                                  write(
                                    withAccount(paths.reuse, account.label),
                                    "DELETE",
                                  ),
                                )
                              : setAccountConfirm({ action: "reuse", account })
                          }
                        />
                        Agent Keys
                      </label>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 px-2 text-xs"
                        disabled={busy !== null}
                        aria-label={`Disconnect ${account.label} for ${name}`}
                        onClick={() =>
                          setAccountConfirm({ action: "disconnect", account })
                        }
                      >
                        Disconnect
                      </Button>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </td>
      <td className={SERVICE_CELL.actions}>
        <div className="flex flex-wrap items-center gap-1 max-sm:justify-start sm:justify-end">
          {busyNote ? (
            <BusyNote>{busyNote}</BusyNote>
          ) : canConnect ? (
            <>
              <Button
                size="sm"
                variant="outline"
                disabled={busy !== null}
                aria-label={`Connect ${oauth.display_name} for Agent Keys for ${name}`}
                onClick={() =>
                  !reuseEnabled
                    ? setConfirm("reuse")
                    : asksAboutSharedAccount
                      ? setConfirm("shared")
                      : connect()
                }
              >
                {busy === "connect" ? "Connecting…" : "Connect"}
              </Button>
              {busy === "connect" && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    operation.current?.abort();
                    setBusy(null);
                  }}
                >
                  Cancel
                </Button>
              )}
            </>
          ) : oauth?.reset_required && canResetAccount ? (
            <Button
              size="sm"
              variant="outline"
              disabled={busy !== null}
              aria-label={`Reset connection for ${name}`}
              onClick={() => setConfirm("account")}
            >
              Reset connection
            </Button>
          ) : canUseAccount ? (
            <Button
              size="sm"
              variant="outline"
              disabled={busy !== null}
              aria-label={`Use account for ${name}`}
              onClick={() => setConfirm("reuse")}
            >
              Use account
            </Button>
          ) : (
            setKeyFirst && (
              <Button
                size="sm"
                variant="outline"
                disabled={busy !== null}
                aria-label={`Set key for ${name}`}
                onClick={openKeyForm}
              >
                Set key
              </Button>
            )
          )}
          <RowActionsMenu
            service={name}
            actions={actions}
            disabled={busy !== null || limitActions.busy}
          />
        </div>
        {oauth && newAccount !== null && (
          <form
            className="mt-2 flex flex-wrap items-center gap-2 sm:justify-end"
            onSubmit={(event) => {
              event.preventDefault();
              if (newAccount && newAccountProblem === null)
                connectAccount(newAccount);
            }}
          >
            <Input
              aria-label={`Label of the other ${oauth.display_name} account`}
              placeholder="Label, such as work"
              autoFocus
              className="h-9 min-w-0 basis-full"
              value={newAccount}
              disabled={busy !== null}
              onChange={(event) => setNewAccount(event.target.value)}
            />
            {newAccountProblem && (
              <p className="basis-full text-xs text-destructive">
                {newAccountProblem}
              </p>
            )}
            <Button
              type="submit"
              size="sm"
              aria-label={`Connect the other ${oauth.display_name} account for ${name}`}
              disabled={
                busy !== null || newAccount === "" || newAccountProblem !== null
              }
            >
              {busy === "connect" ? "Connecting…" : "Connect"}
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => {
                operation.current?.abort();
                setBusy(null);
                setNewAccount(null);
              }}
            >
              Cancel
            </Button>
          </form>
        )}
        {canManageKey && editing && (
          <form
            className="mt-2 flex flex-wrap items-center gap-2 sm:justify-end"
            onSubmit={save}
          >
            <Input
              type="password"
              autoComplete="off"
              aria-label={keyLabel}
              placeholder="Paste the key"
              autoFocus
              className="h-9 min-w-0 basis-full"
              value={secret}
              disabled={busy !== null}
              onChange={(event) => setSecret(event.target.value)}
            />
            <Button
              type="submit"
              size="sm"
              disabled={busy !== null || !secret.trim()}
            >
              {busy === "save" ? (
                <>
                  <Loader2
                    className="mr-2 h-3.5 w-3.5 animate-spin"
                    aria-hidden="true"
                  />
                  Saving…
                </>
              ) : (
                "Save"
              )}
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={busy !== null}
              onClick={closeEditor}
            >
              Cancel
            </Button>
          </form>
        )}
        {error && (
          <Alert variant="destructive" className="mt-2 p-2 text-left">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {limitActions.dialog}
        <ConfirmDialog
          open={confirm === "key"}
          title={`Remove ${keyLabel}?`}
          description={removeWarning(service)}
          confirmLabel="Remove"
          onCancel={() => setConfirm(null)}
          onConfirm={() =>
            void run(
              "remove",
              "Could not remove the key. Try again.",
              write(paths.secret, "DELETE"),
            )
          }
        />
        {editable && (
          <ConfirmDialog
            open={confirm === "service"}
            title={`Delete ${service.display_name}?`}
            description={serviceEditing.deleteWarning(service)}
            confirmLabel="Delete"
            onCancel={() => setConfirm(null)}
            onConfirm={() => void deleteService()}
          />
        )}
        {oauth && (
          <SharedAccountDialog
            open={confirm === "shared"}
            agent={agent.displayName}
            provider={oauth.display_name}
            onCancel={() => setConfirm(null)}
            onContinue={() => {
              setConfirm(null);
              connect();
            }}
          />
        )}
        {oauth && (
          <AccountReuseDialog
            open={confirm === "reuse"}
            provider={oauth.display_name}
            description={accountReuseDescription({
              provider: oauth.display_name,
              account: oauth.connected
                ? (oauth.account_label ??
                  `the connected ${oauth.display_name} account`)
                : `the ${oauth.display_name} account you connect`,
              service: name,
              sharedAgent: asksAboutSharedAccount ? agent.displayName : null,
              unscoped: agent.scope === "unscoped",
            })}
            confirmLabel={oauth.connected ? "Use account" : "Continue"}
            onCancel={() => setConfirm(null)}
            onContinue={() =>
              oauth.connected
                ? void run(
                    "reuse",
                    "Could not allow the account. Try again.",
                    write(paths.reuse, "PUT", REUSE_BODY),
                  )
                : connect()
            }
          />
        )}
        {oauth && (
          <AccountReuseDialog
            open={accountConfirm?.action === "reuse"}
            provider={oauth.display_name}
            description={accountReuseDescription({
              provider: oauth.display_name,
              account:
                accountConfirm?.account.account_label ??
                `the ${accountConfirm?.account.label} account`,
              service: name,
              sharedAgent: asksAboutSharedAccount ? agent.displayName : null,
              unscoped: agent.scope === "unscoped",
            })}
            confirmLabel="Use account"
            onCancel={() => setAccountConfirm(null)}
            onContinue={() =>
              accountConfirm &&
              void run(
                "reuse",
                "Could not allow the account. Try again.",
                write(paths.reuse, "PUT", {
                  ...REUSE_BODY,
                  account: accountConfirm.account.label,
                }),
              )
            }
          />
        )}
        {oauth && (
          <ConfirmDialog
            open={accountConfirm?.action === "disconnect"}
            title={`Disconnect ${accountConfirm?.account.label}?`}
            description={`This removes the ${oauth.display_name} account saved as ${accountConfirm?.account.label} for ${agent.displayName}, and Agent Keys stops using it.`}
            confirmLabel="Disconnect"
            onCancel={() => setAccountConfirm(null)}
            onConfirm={() =>
              accountConfirm &&
              void run(
                "disconnect",
                "Could not disconnect. Try again.",
                write(paths.disconnect, "POST", {
                  account: accountConfirm.account.label,
                }),
              )
            }
          />
        )}
        {oauth && (
          <ConfirmDialog
            open={confirm === "account"}
            title={`${oauth.connected ? "Disconnect" : "Reset"} ${oauth.display_name}?`}
            description={egressDisconnectDescription(
              oauth.display_name,
              agent.displayName,
              servicesUsingAccount(agent.services, oauth.provider),
              reuseEnabled,
            )}
            confirmLabel={oauth.connected ? "Disconnect" : "Reset connection"}
            onCancel={() => setConfirm(null)}
            onConfirm={() =>
              void run(
                "disconnect",
                "Could not disconnect. Try again.",
                write(paths.disconnect, "POST"),
              )
            }
          />
        )}
      </td>
    </tr>
  );
}
