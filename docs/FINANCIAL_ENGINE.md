# FPO DPR — How the Financial Engine Works

*A plain-language guide to every number in the Detailed Project Report, with the exact formulas used.*

## Overview

Every figure in a DPR is produced by a **deterministic financial engine** — ordinary arithmetic, no AI. The language model only writes the prose; it never touches a number. This means the same inputs always give the same figures, and every figure can be traced back to a formula and an input.

**Conventions used throughout:**

- All money is in **Rs. Lakh** (1 Lakh = Rs. 100,000). Rupee inputs are divided by 100,000.
- Projections run for **5 years** (Year 1 … Year 5).
- Two kinds of input drive the engine:
  - **Form inputs** — what the FPO enters (capacity, prices, capex, funding, scheme, location).
  - **Assumptions (config)** — standardised financial parameters per business line (interest, escalation, depreciation, tax, working-capital norms, wage tables).
- Two revenue models:
  - **Processing** — raw material is converted into a product (mill, oil, juice, feed…).
  - **Service** — capacity is rented out; there is no raw-material-to-product conversion (warehouse, custom hiring).

The engine also enforces one hard check: **Means of Finance must exactly equal the Total Project Cost.** If they differ, the report is rejected.

---

## 1. Production & Sales

**What it does:** works out how much is produced and sold each year, and the money value of sales and raw material.

**Capacity ramp-up.** The unit does not run at 100% from day one; utilisation ramps up over the 5 years (e.g. 75%, 80%, 85%, 90%, 90%). Let $u_y$ be the utilisation in year $y$.

**Price escalation.** Selling and raw-material rates rise each year by the escalation rate $g$ (Year 1 uses the exact base price):

$$\text{price}_y = \text{price}_{\text{base}} \times (1+g)^{\,y}$$

**Processing model** — annual throughput:

$$\text{RM requirement}_{100\%} = \text{capacity per hour} \times \text{shift hours} \times \text{working days}$$

$$\text{Output}_{100\%} = \text{RM requirement}_{100\%} \times (1 - \text{wastage})$$

$$\text{RM used}_y = \text{RM requirement}_{100\%}\times u_y, \qquad \text{Production}_y = \text{Output}_{100\%}\times u_y$$

A small amount of finished goods is held as closing stock, so **net sales quantity** = production + opening stock − closing stock:

$$\text{Closing stock}_y = \frac{\text{Production}_y}{\text{working days}}\times \text{FG days}, \qquad \text{Net sales}_y = \text{Production}_y + \text{Open}_y - \text{Close}_y$$

**Service model** (warehouse / hiring): capacity is rented service units, no wastage, no stock:

$$\text{Net sales}_y = \text{rated units}\times u_y$$

**Money values:**

$$\text{Sales}_y = \frac{\text{Net sales}_y \times \text{sale price}_y}{100{,}000}$$

Raw-material cost is computed one of two ways (form choice):

$$\text{RM cost}_y = \frac{\text{RM used}_y \times \text{RM rate}_y}{100{,}000} \quad\text{(per-unit)}, \qquad \text{RM cost}_y = \text{Sales}_y \times f \quad\text{(\% of revenue)}$$

---

## 2. Fixed Capital (one-time capital cost)

**What it does:** adds up the permanent assets.

$$\text{Fixed Capital} = \text{Machinery} + \text{Misc. Fixed Assets} + \text{Civil Construction}$$

**Land treatment** (from the form's land status):

| Land status | Civil construction | Rent |
|---|---|---|
| Owned | counted in capital | none |
| To be acquired | counted in capital | none |
| Leased | *(currently set to 0)* | rent charged to P&L = monthly rent × 12 |

> *Note: for a leased warehouse where the FPO constructs its own building, civil construction should be capitalised — this is a known refinement in progress.*

---

## 3. Depreciation (Written-Down Value method)

**What it does:** spreads the cost of assets over their life. Each year, a fixed percentage of the *remaining* value is charged.

$$\text{Depreciation}_y = \text{Plant}_y \times r_{pm} + \text{Misc}_y \times r_{misc}$$

$$\text{Plant}_{y+1} = \text{Plant}_y \times (1 - r_{pm}) \qquad(\text{same for Misc})$$

where Plant = machinery + civil, and $r_{pm}, r_{misc}$ are the depreciation rates (e.g. 10%). The remaining value each year is the **Net Block**.

---

## 4. Operating Costs (yearly running costs)

**Labour, staff and power** are "semi-variable" — they scale with utilisation and escalate slowly:

$$\text{Cost}_y = \text{base} \times \text{factor}_y \times (1 + g_{\text{semi}})^{\,y}$$

where $\text{factor}_y = u_y$ if labour scales with utilisation, otherwise $u_y / u_1$ (full roster from Year 1). Bases:

$$\text{Labour base} = \frac{\sum(\text{count}\times\text{monthly wage})\times 12}{100{,}000}, \qquad \text{Power base} = \frac{\text{units/day}\times\text{days}\times\text{rate}}{100{,}000}$$

**Sales-linked costs** (percentages of sales, set per business line):

$$\text{Repair}_y = \text{Sales}_y \times r_{\text{rep}}, \quad \text{Packaging}_y = \text{Sales}_y \times r_{\text{pack}}, \quad \text{Selling \& Admin}_y = \text{Sales}_y \times r_{\text{sell}}$$

**Rent** (leased premises only): $\text{Rent}_y = \text{Rent base} \times (1+g_{\text{rent}})^{\,y}$

**Cost of Production (COP):**

$$\text{COP}_y = \text{RM cost}_y + \text{Power}_y + \text{Depreciation}_y + \text{Labour}_y + \text{Repair}_y + \text{Packaging}_y$$

---

## 5. Government Scheme & Means of Finance

**What it does:** decides how the project is paid for. The scheme (form choice) sets the split. Eligible cost = fixed capital.

**Capital-subsidy schemes** (PM-FME: 35%, cap Rs. 10 L; MIDH: 40%):

$$\text{Subsidy} = \min(\text{subsidy}\%\times\text{Fixed Capital},\ \text{ceiling})$$
$$\text{Promoter (fixed)} = \text{min contribution}\% \times \text{Fixed Capital}$$
$$\text{Term Loan} = \text{Fixed Capital} - \text{Subsidy} - \text{Promoter (fixed)}$$

**Interest-subvention scheme** (AIF): no grant, but the loan interest rate is reduced:

$$\text{Term Loan} = \text{Fixed Capital} - \text{Promoter (fixed)}, \qquad \text{Effective rate} = \text{Base rate} - \text{subvention}$$

**Self-funded:** $\text{Term Loan} = \text{Fixed Capital} - \text{Promoter equity committed}$ (no subsidy).

| Scheme | Type | Key numbers |
|---|---|---|
| PM-FME | Capital subsidy | 35% of fixed capital, capped at Rs. 10 L; 10% promoter |
| MIDH | Capital subsidy | 40% of fixed capital; 10% promoter |
| AIF | Interest subvention | 3% rate rebate, up to Rs. 200 L loan, 7 years |
| Self-funded | — | promoter equity + term loan, no subsidy |

---

## 6. Term Loan Repayment Schedule

**What it does:** works out yearly principal and interest. Equal monthly principal after a moratorium (interest-only) period.

$$\text{Monthly principal} = \frac{\text{Loan amount}}{\text{tenor months} - \text{moratorium months}}$$

Each month: interest = outstanding balance × (rate ÷ 12); principal is paid only after the moratorium. These are summed per year to give **Principal Repaid**, **Interest**, and **Closing Balance**.

---

## 7. Working Capital (traditional method)

**What it does:** the running-cash cushion needed before customers pay. Built from stock + receivables, less what suppliers let you defer.

$$\text{RM stock} = \frac{\text{RM cost}_y}{\text{days}}\times\text{RM days}, \quad \text{Debtors} = \frac{\text{Sales}_y}{\text{days}}\times\text{receivable days}, \quad \text{Creditors} = \frac{\text{RM cost}_y}{\text{days}}\times\text{payable days}$$

$$\text{Gross Working Capital} = (\text{RM stock} + \text{FG stock} - \text{Creditors}) + \text{Debtors}$$

It is funded by a promoter **margin** and a bank **cash-credit (CC) loan**:

$$\text{WC Margin} = \text{GWC}\times m, \qquad \text{WC Bank Loan (CC)} = \text{GWC}\times(1-m)$$

where $m$ is the margin percentage (e.g. 25%). WC interest is charged on the Year-1 sanctioned bank limit.

---

## 8. Total Project Cost & Financing

$$\text{Total Project Cost} = \text{Fixed Capital} + \text{Gross Working Capital (Year 1)}$$

$$\text{Promoter's Contribution} = \text{Promoter (fixed)} + \text{WC Margin}$$

**Means of Finance = Promoter's Contribution + Subsidy + Term Loan + WC Bank Loan**, and this is guaranteed to equal the Total Project Cost.

---

## 9. Profitability (Profit & Loss)

$$\text{Cost of Sales}_y = \text{COP}_y + \text{Opening FG} - \text{Closing FG}$$
$$\text{Gross Profit}_y = \text{Sales}_y - \text{Cost of Sales}_y$$
$$\text{Profit Before Tax}_y = \text{Gross Profit}_y - (\text{Staff} + \text{Term-loan interest} + \text{WC interest} + \text{Rent} + \text{Selling \& Admin})$$
$$\text{Tax}_y = \max(\text{PBT}_y - \text{threshold},\ 0)\times \text{tax rate}$$
$$\text{Profit After Tax}_y = \text{PBT}_y - \text{Tax}_y, \qquad \text{Cash Accrual}_y = \text{PAT}_y + \text{Depreciation}_y$$

---

## 10. Break-Even Point

Costs are split into variable (move with sales) and fixed (do not). The break-even is the % of capacity at which the business just covers its costs.

$$\text{Contribution}_y = \text{Sales}_y - \text{Variable Cost}_y, \qquad \text{Break-Even \%} = \frac{\text{Fixed Cost}_y}{\text{Contribution}_y}\times 100$$

*(Power and labour are split part-variable, part-fixed; depreciation, term-loan interest and rent are fully fixed.)*

---

## 11. Balance Sheet, Cash Flow & DSCR

**Net Worth** grows as profits accumulate:
$$\text{Net Worth}_y = \text{Promoter Equity} + \text{Subsidy} + \sum_{k \le y}\text{PAT}_k$$

**Closing Cash** accumulates cash accrual less loan principal:
$$\text{Closing Cash}_y = \text{Closing Cash}_{y-1} + \text{Cash Accrual}_y - \text{Principal Repaid}_y$$

**Debt-Service Coverage Ratio (DSCR)** — the key bankability metric; how comfortably the business covers its loan payments:
$$\text{DSCR}_y = \frac{\text{Cash Accrual}_y + \text{Term-loan interest}_y}{\text{Principal}_y + \text{Term-loan interest}_y}$$

The **Average DSCR** (mean over the loan years) must clear the target minimum (typically 1.5) for the project to be bankable.

---

## In one sentence

The FPO provides the business facts, the template provides the standardised assumptions, and this engine turns them — through the formulas above — into every table in the DPR, with the guarantee that the financing always balances the cost and the language model never invents a figure.
