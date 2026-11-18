# Generation evaluation notes

Last updated: 2026-11-16

This file records failures observed during manual testing of the local generation model. These cases must become regression tests before the assistant is considered safe or reliable.

## 1. Compensation information and access control

### Test question

> Combien le salaire d'un manager ?

### Observed answer

> Le salaire d'un manager n'est pas explicitement mentionné dans les sources fournies.

### Risk

The answer is appropriate for the current corpus because no manager salary is available. However, this test exposes a future confidentiality risk: if salary or other restricted employee information is added to the index, an unrestricted retrieval request could expose it to every user.

A prompt instruction is not an authorization mechanism. Confidentiality must be enforced before retrieved chunks are sent to the model.

### Required improvement

- Add document and chunk access metadata when Batch 3 permissions are implemented.
- Authenticate the requesting user and resolve their role before retrieval.
- Apply mandatory Qdrant filters so unauthorized chunks never enter the generation context.
- Return a neutral access-denied response without confirming whether restricted information exists.
- Add regression tests for salary, compensation, individual employee data, and restricted HR documents across multiple roles.

### Expected behavior

For a user without the required permission, the assistant must not reveal the value, source, or existence of restricted compensation information. It should direct the user to the authorized HR channel.

## 2. Placeholder answer returned for sickness during leave

### Gold question

> Je suis tombé malade pendant mes vacances. Est-ce que je récupère mes jours ?

Gold id: `Q016`

### Observed answer

> A supported answer.

### Problem

The response is a prompt placeholder rather than an answer grounded in the retrieved policy. A likely cause is example leakage from the JSON output example in the system prompt, which currently contains the same placeholder text.

### Expected answer

The assistant should explain that the affected days can be converted to sick leave and credited back to the annual-leave balance, provided that a medical certificate is sent to the HR team within 48 hours.

### Required improvement

- Remove semantic placeholder text such as `A supported answer.` from the prompt example.
- Prefer a strict response schema that describes fields without supplying reusable answer content.
- Add a validator or regression assertion that rejects known placeholder answers.
- Re-run Q016 with its expected source and verify factual completeness, refusal behavior, and citations.
- Check the remaining gold questions for prompt-example leakage.

## Acceptance criteria

- Unauthorized users cannot retrieve or generate restricted compensation information.
- Q016 produces a factual French answer containing the medical-certificate requirement and the 48-hour deadline.
- The generated answer cites only sources supplied in the context.
- No response contains prompt placeholders or schema-example text.
