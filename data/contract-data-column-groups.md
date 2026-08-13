# Contract Data Column Groups

This SQL file stores contract and approval workflow data for a business form table. It combines contract details, workflow state, approval routing, and system metadata for each record.

## 1) Core contract identity

These columns identify the contract itself and its lifecycle basics.

Typical information in this group includes:

- internal record ID
- contract title or reference number
- counterparty name
- contract start and end dates
- whether the record reflects a final or signed version

### Purpose

This group is used to uniquely identify each contract record, support searching and filtering, and track the agreement across its lifecycle.

## 2) Contract content and commercial terms

These columns describe what the contract is about and the main commercial terms.

Typical information in this group includes:

- product or service description
- contract amount
- notes explaining the amount
- justification fields for threshold-based questions
- additional context about the deal

### Purpose

This group explains the commercial scope of the agreement and supports financial and business review.

## 3) Drafting and document availability

These columns track whether the contract is still in draft form and whether the supporting documents exist.

Typical information in this group includes:

- draft contract indicator
- signed contract indicator
- final version indicator
- upload or attachment-related flags

### Purpose

This group shows where the contract is in the document lifecycle and whether it is ready for review or execution.

## 4) Risk, threshold, and policy checks

A large portion of the file contains workflow flags connected to policy and compliance.

Typical information in this group includes:

- whether the amount exceeds a threshold
- whether liabilities or guarantees are involved
- whether the contract has an end date
- whether legal or finance approval is needed
- whether related-party or data-related risks exist

### Purpose

These fields help enforce internal policy and determine which review steps are required.

## 5) Approval routing and sign-off levels

These columns store how the contract moves through the approval process.

Typical information in this group includes:

- business approval level
- finance approval level
- legal approval flags
- group finance or committee approval flags
- approver role and level fields
- final approval or sign-off status

### Purpose

This group tracks the approval chain, identifies which teams reviewed the contract, and records whether the contract is approved, pending, or blocked.

## 6) Organizational ownership and routing

These columns indicate who owns the contract internally and which teams are responsible for it.

Typical information in this group includes:

- requestor
- business unit
- department
- contract owner
- approver groups
- entity finance head
- group CFO or management committee references

### Purpose

This group routes the contract to the right internal stakeholders, establishes accountability, and supports reporting by business area.

## 7) Workflow status and system-generated state

These fields relate more to the workflow engine and application state than to the legal contract itself.

Typical information in this group includes:

- status
- saved or submitted flags
- draft or final visibility flags
- review tier
- mode or form mode identifiers
- delete or process flags

### Purpose

These columns control the record lifecycle in the application, including whether the record is editable, visible, submitted, or archived.

## 8) Metadata and audit trail

These columns capture who created or modified the record and when.

Typical information in this group includes:

- creator information
- creation date and time
- modifier information
- modification date and time
- UUIDs
- process IDs
- external/internal record mappings

### Purpose

This group provides auditability, traceability, and support for debugging or synchronization.

## 9) Conditional prompts and UI control fields

A number of columns appear to drive the user interface and conditional form behavior.

Typical information in this group includes:

- warning prompts
- show/hide controls for form sections
- justification prompts
- conditional approval questions
- visibility rules for certain fields

### Purpose

These fields make the form dynamic so that only relevant prompts and sections appear based on the contract’s type, amount, or risk profile.

## Overall interpretation

In plain language, the SQL stores a mix of:

- contract master data
- approval workflow state
- risk and compliance checks
- team ownership and routing
- audit metadata
- UI control flags

It is more than a list of contracts. It is a complete workflow record for each contract, showing what the contract is, who owns it, how much it is worth, which policy checks apply, who approved it, and how the system should display it.