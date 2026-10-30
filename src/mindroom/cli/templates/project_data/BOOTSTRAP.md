# BOOTSTRAP.md - First Run

First check whether setup already happened: `IDENTITY.md` names operators, or `memory/sources.md` or `memory/decisions.md` has rows beyond the empty seed tables.
If it has, a later seeding copy brought this file back, so delete it without rerunning onboarding or changing any other file, and carry on with the request at hand.
Only a fresh workspace goes through the checklist below.

Work through this checklist once, in the project room, then delete this file.
When a step needs someone who is not around, log it in `memory/questions.md`, ask that person in the room, and carry on with the next step.
A step waiting on a logged question does not block the rest.

1. **Operators.**
   Ask who the operators are and record their Matrix IDs in `IDENTITY.md`.
2. **Scope and precedence.**
   Ask the operators for the agent's name, the project and its goal, what is in and out of scope, and which source wins when sources disagree.
   Record the answers in `IDENTITY.md`.
3. **Probe each tool and knowledge base.**
   Run one harmless read with each, such as a listing or a search for the project name.
   Fill the integration table in `TOOLS.md` with the account each one uses and what it may read and write.
   Log anything that failed because of the sandbox in `memory/sandbox-friction.md`.
4. **Source register.**
   Add every source you found or were told about to `memory/sources.md`, with its access state, authority class, and today's date.
5. **One cited answer.**
   Answer one real question about the project from its sources, with citations, so people can check how you work.
6. **Recurring digest.**
   Offer a recurring digest built with the `status-report` skill.
   Set it up only once the room has agreed on everything `AGENTS.md` requires for a schedule.
7. **Placeholders.**
   Replace every placeholder an operator has confirmed, and make sure a question is logged for each one that remains.
8. **Finish.**
   Tell the room what is set up and what is still open, then delete this file.
