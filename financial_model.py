"""
Rental Investment Model — Phase 1
Supports: Short-Term Rental (STR) and Long-Term Rental (LTR)
Markets: Indianapolis, Los Angeles, Las Vegas
Usage: python financial_model.py
"""

# ── Market configs ────────────────────────────────────────────────────────────
# ADR and occupancy are placeholders until Phase 2 pulls real AirDNA data.
# Tax rates and fees are current as of 2024.

# ── Property type cost profiles ──────────────────────────────────────────────
# Estimates based on typical STR operating costs by size.
# Override any value per-deal using the override args in analyze_deal().
# TODO (Phase 2 upgrade): replace with Thumbtack/Angi scraped cost data by market.

PROPERTY_TYPES = {
    "studio":   {"cleaning_per_turnover": 65,  "supplies_per_night": 5,  "utilities_annual": 1800, "maintenance_pct": 0.008},
    "1br":      {"cleaning_per_turnover": 85,  "supplies_per_night": 7,  "utilities_annual": 2000, "maintenance_pct": 0.009},
    "2br":      {"cleaning_per_turnover": 110, "supplies_per_night": 10, "utilities_annual": 2400, "maintenance_pct": 0.010},
    "3br":      {"cleaning_per_turnover": 145, "supplies_per_night": 14, "utilities_annual": 2800, "maintenance_pct": 0.011},
    "4br_plus": {"cleaning_per_turnover": 185, "supplies_per_night": 18, "utilities_annual": 3400, "maintenance_pct": 0.012},
}

MARKETS = {
    "indianapolis": {
        "label": "Indianapolis, IN",
        "property_tax_rate": 0.0085,
        "transient_occupancy_tax": 0.10,
        "avg_mgmt_fee": 0.20,            # STR mgmt fee
        "avg_adr": 135,
        "avg_occupancy": 0.62,
        "str_status": "friendly",
        "insurance_annual": 2200,        # STR insurance
        "notes": "No state preemption issues. STR-friendly market.",
        # LTR fields
        "ltr_avg_rent_2br": 1350,        # avg monthly rent for 2BR (placeholder)
        "ltr_vacancy_rate": 0.07,        # ~1 month vacancy/yr
        "ltr_mgmt_fee": 0.09,            # LTR mgmt fee % of rent
        "ltr_insurance_annual": 1400,    # standard landlord insurance
    },
    "los_angeles": {
        "label": "Los Angeles, CA",
        "property_tax_rate": 0.0125,
        "transient_occupancy_tax": 0.145,
        "avg_mgmt_fee": 0.25,
        "avg_adr": 210,
        "avg_occupancy": 0.58,
        "str_status": "restricted",
        "insurance_annual": 3800,
        "notes": "Primary residence only. 120-night/yr cap. High reg risk.",
        # LTR fields
        "ltr_avg_rent_2br": 3200,
        "ltr_vacancy_rate": 0.05,
        "ltr_mgmt_fee": 0.08,
        "ltr_insurance_annual": 2400,
    },
    "las_vegas": {
        "label": "Las Vegas, NV",
        "property_tax_rate": 0.007,
        "transient_occupancy_tax": 0.134,
        "avg_mgmt_fee": 0.22,
        "avg_adr": 175,
        "avg_occupancy": 0.66,
        "str_status": "permit_required",
        "insurance_annual": 2600,
        "notes": "License required (~$500/yr). No owner-occupancy requirement.",
        # LTR fields
        "ltr_avg_rent_2br": 1650,
        "ltr_vacancy_rate": 0.06,
        "ltr_mgmt_fee": 0.09,
        "ltr_insurance_annual": 1600,
    },
}

# ── Mortgage calculator ───────────────────────────────────────────────────────

def monthly_mortgage(purchase_price, down_pct, annual_rate, term_years=30):
    principal = purchase_price * (1 - down_pct)
    r = annual_rate / 12
    n = term_years * 12
    if r == 0:
        return principal / n
    payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return payment

# ── Core financial model ──────────────────────────────────────────────────────

def analyze_deal(
    market_key,
    purchase_price,
    down_pct=0.20,
    annual_rate=0.0725,
    closing_cost_pct=0.03,
    renovation=0,
    hoa_monthly=0,
    term_years=30,
    # ── Rental type ────────────────────────────────────────────────────────
    rental_type="str",             # "str" or "ltr"
    # ── STR-specific ───────────────────────────────────────────────────────
    custom_adr=None,
    custom_occupancy=None,
    property_type="2br",           # studio | 1br | 2br | 3br | 4br_plus
    override_cleaning=None,        # $ per turnover
    override_supplies=None,        # $ per occupied night
    # ── LTR-specific ───────────────────────────────────────────────────────
    monthly_rent=None,             # actual rent; falls back to market default
    # ── Shared overrides ───────────────────────────────────────────────────
    override_maintenance=None,     # $ annual
    override_utilities=None,       # $ annual
):
    market  = MARKETS[market_key]
    profile = PROPERTY_TYPES.get(property_type, PROPERTY_TYPES["2br"])

    down_payment   = purchase_price * down_pct
    closing_costs  = purchase_price * closing_cost_pct
    total_invested = down_payment + closing_costs + renovation
    hoa            = hoa_monthly * 12
    property_tax   = purchase_price * market["property_tax_rate"]
    maintenance    = override_maintenance or purchase_price * profile["maintenance_pct"]
    monthly_pmt    = monthly_mortgage(purchase_price, down_pct, annual_rate, term_years)
    annual_debt    = monthly_pmt * 12

    divider = "─" * 52

    # ══════════════════════════════════════════════════════════════════════
    if rental_type == "ltr":
        # ── LTR Revenue ──────────────────────────────────────────────────
        rent          = monthly_rent or market["ltr_avg_rent_2br"]
        gross_revenue = rent * 12
        vacancy_loss  = gross_revenue * market["ltr_vacancy_rate"]
        eff_income    = gross_revenue - vacancy_loss

        # ── LTR Expenses ─────────────────────────────────────────────────
        # No TOT, no cleaning turnovers, no platform fees, no supplies
        # Tenant pays utilities in most LTR arrangements
        mgmt_fee  = eff_income * market["ltr_mgmt_fee"]
        insurance = market["ltr_insurance_annual"]
        utilities = override_utilities or 0    # tenant-paid by default

        total_expenses = mgmt_fee + property_tax + insurance + maintenance + utilities + hoa
        noi            = eff_income - total_expenses

        annual_cf  = noi - annual_debt
        monthly_cf = annual_cf / 12
        cap_rate   = (noi / purchase_price) * 100
        coc_return = (annual_cf / total_invested) * 100
        gross_yield = (gross_revenue / purchase_price) * 100
        gross_rent_mult = purchase_price / gross_revenue  # GRM — lower is better

        # ── LTR Output ───────────────────────────────────────────────────
        print(f"\n{'═' * 52}")
        print(f"  {market['label']}  |  ${purchase_price:,.0f}  |  LTR  |  {property_type}")
        print(f"{'═' * 52}")

        print(f"\n  DEAL INPUTS")
        print(f"  {divider}")
        print(f"  Purchase price      ${purchase_price:>12,.0f}")
        print(f"  Down payment        {down_pct*100:>11.0f}%  (${down_payment:,.0f})")
        print(f"  Interest rate       {annual_rate*100:>11.2f}%")
        print(f"  Closing costs       ${closing_cost_pct*100:>10.0f}%  (${closing_costs:,.0f})")
        print(f"  Renovation          ${renovation:>12,.0f}")
        print(f"  Total cash in       ${total_invested:>12,.0f}")

        print(f"\n  REVENUE (annual)")
        print(f"  {divider}")
        rent_src = "actual" if monthly_rent else "market avg"
        print(f"  Monthly rent        ${rent:>12,.0f}/mo  ({rent_src})")
        print(f"  Gross revenue       ${gross_revenue:>12,.0f}")
        print(f"  Vacancy loss        -${vacancy_loss:>11,.0f}  ({market['ltr_vacancy_rate']*100:.0f}%)")
        print(f"  Effective income    ${eff_income:>12,.0f}")

        print(f"\n  EXPENSES (annual)")
        print(f"  {divider}")
        print(f"  Mgmt fee ({market['ltr_mgmt_fee']*100:.0f}%)      ${mgmt_fee:>12,.0f}")
        print(f"  Property tax        ${property_tax:>12,.0f}")
        print(f"  Insurance           ${insurance:>12,.0f}")
        print(f"  Maintenance         ${maintenance:>12,.0f}  ({'actual' if override_maintenance else 'estimate'})")
        print(f"  Utilities           ${utilities:>12,.0f}  (tenant-paid)" if not override_utilities else f"  Utilities           ${utilities:>12,.0f}  (landlord-paid)")
        print(f"  HOA                 ${hoa:>12,.0f}")
        print(f"  {divider}")
        print(f"  Total expenses      ${total_expenses:>12,.0f}")
        print(f"  NOI                 ${noi:>12,.0f}")

        print(f"\n  FINANCING (annual)")
        print(f"  {divider}")
        print(f"  Mortgage payment    ${monthly_pmt:>12,.0f}/mo  (${annual_debt:,.0f}/yr)")
        print(f"  {divider}")

        cf_label = "  ✓ CASH FLOW" if annual_cf >= 0 else "  ✗ CASH FLOW"
        print(f"\n  RETURNS")
        print(f"  {divider}")
        print(f"{cf_label}         ${monthly_cf:>+12,.0f}/mo  (${annual_cf:+,.0f}/yr)")
        print(f"  Cap rate            {cap_rate:>11.2f}%")
        print(f"  Cash-on-cash        {coc_return:>11.2f}%")
        print(f"  Gross yield         {gross_yield:>11.2f}%")
        print(f"  Gross rent mult     {gross_rent_mult:>11.1f}x  (target <15x)")
        print(f"{'═' * 52}\n")

        return {
            "market": market_key, "rental_type": "ltr",
            "purchase_price": purchase_price, "monthly_cash_flow": monthly_cf,
            "annual_cash_flow": annual_cf, "cap_rate": cap_rate,
            "coc_return": coc_return, "gross_yield": gross_yield,
            "total_invested": total_invested, "noi": noi,
        }

    # ══════════════════════════════════════════════════════════════════════
    else:  # STR
        # ── STR Revenue ──────────────────────────────────────────────────
        adr       = custom_adr or market["avg_adr"]
        occupancy = custom_occupancy or market["avg_occupancy"]

        if market_key == "los_angeles":
            occupied_nights = min(365 * occupancy, 120)
        else:
            occupied_nights = 365 * occupancy

        gross_revenue = adr * occupied_nights
        tot           = gross_revenue * market["transient_occupancy_tax"]
        net_revenue   = gross_revenue - tot

        # ── STR Expenses ─────────────────────────────────────────────────
        cleaning_per_turnover = override_cleaning or profile["cleaning_per_turnover"]
        supplies_per_night    = override_supplies  or profile["supplies_per_night"]
        utilities             = override_utilities or profile["utilities_annual"]
        turnovers             = occupied_nights / 3.5
        cleaning              = cleaning_per_turnover * turnovers

        mgmt_fee      = net_revenue * market["avg_mgmt_fee"]
        insurance     = market["insurance_annual"]
        supplies      = supplies_per_night * occupied_nights
        platform_fees = gross_revenue * 0.03

        total_expenses = (
            mgmt_fee + property_tax + insurance + maintenance +
            utilities + supplies + cleaning + hoa + platform_fees
        )
        noi = net_revenue - total_expenses

        annual_cf   = noi - annual_debt
        monthly_cf  = annual_cf / 12
        cap_rate    = (noi / purchase_price) * 100
        coc_return  = (annual_cf / total_invested) * 100
        gross_yield = (gross_revenue / purchase_price) * 100
        break_even  = (total_expenses + annual_debt) / (adr * 365 * (1 - market["transient_occupancy_tax"])) * 100

        # ── STR Output ───────────────────────────────────────────────────
        print(f"\n{'═' * 52}")
        print(f"  {market['label']}  |  ${purchase_price:,.0f}  |  STR  |  {property_type}")
        print(f"{'═' * 52}")

        print(f"\n  DEAL INPUTS")
        print(f"  {divider}")
        print(f"  Purchase price      ${purchase_price:>12,.0f}")
        print(f"  Down payment        {down_pct*100:>11.0f}%  (${down_payment:,.0f})")
        print(f"  Interest rate       {annual_rate*100:>11.2f}%")
        print(f"  Closing costs       ${closing_cost_pct*100:>10.0f}%  (${closing_costs:,.0f})")
        print(f"  Renovation          ${renovation:>12,.0f}")
        print(f"  Total cash in       ${total_invested:>12,.0f}")

        print(f"\n  REVENUE (annual)")
        print(f"  {divider}")
        print(f"  ADR                 ${adr:>12,.0f}/night")
        print(f"  Occupancy           {occupancy*100:>11.0f}%  ({occupied_nights:.0f} nights)")
        print(f"  Gross revenue       ${gross_revenue:>12,.0f}")
        print(f"  TOT paid to city   -${tot:>12,.0f}  ({market['transient_occupancy_tax']*100:.1f}%)")
        print(f"  Net revenue         ${net_revenue:>12,.0f}")

        print(f"\n  EXPENSES (annual)")
        print(f"  {divider}")
        src = "actual" if override_cleaning else "estimate"
        print(f"  Mgmt fee            ${mgmt_fee:>12,.0f}")
        print(f"  Property tax        ${property_tax:>12,.0f}")
        print(f"  Insurance           ${insurance:>12,.0f}")
        print(f"  Maintenance         ${maintenance:>12,.0f}  ({'actual' if override_maintenance else 'estimate'})")
        print(f"  Utilities           ${utilities:>12,.0f}  ({'actual' if override_utilities else 'estimate'})")
        print(f"  Cleaning            ${cleaning:>12,.0f}  ({src}, {turnovers:.0f} turnovers)")
        print(f"  Supplies            ${supplies:>12,.0f}  ({'actual' if override_supplies else 'estimate'})")
        print(f"  HOA                 ${hoa:>12,.0f}")
        print(f"  Platform fees (3%)  ${platform_fees:>12,.0f}")
        print(f"  {divider}")
        print(f"  Total expenses      ${total_expenses:>12,.0f}")
        print(f"  NOI                 ${noi:>12,.0f}")

        print(f"\n  FINANCING (annual)")
        print(f"  {divider}")
        print(f"  Mortgage payment    ${monthly_pmt:>12,.0f}/mo  (${annual_debt:,.0f}/yr)")
        print(f"  {divider}")

        cf_label = "  ✓ CASH FLOW" if annual_cf >= 0 else "  ✗ CASH FLOW"
        print(f"\n  RETURNS")
        print(f"  {divider}")
        print(f"{cf_label}         ${monthly_cf:>+12,.0f}/mo  (${annual_cf:+,.0f}/yr)")
        print(f"  Cap rate            {cap_rate:>11.2f}%")
        print(f"  Cash-on-cash        {coc_return:>11.2f}%")
        print(f"  Gross yield         {gross_yield:>11.2f}%")
        print(f"  Break-even occupancy{break_even:>10.1f}%")

        print(f"\n  REGULATORY")
        print(f"  {divider}")
        print(f"  Status              {market['str_status'].upper():>20}")
        print(f"  Note: {market['notes']}")
        print(f"{'═' * 52}\n")

        return {
            "market": market_key, "rental_type": "str",
            "purchase_price": purchase_price, "monthly_cash_flow": monthly_cf,
            "annual_cash_flow": annual_cf, "cap_rate": cap_rate,
            "coc_return": coc_return, "gross_yield": gross_yield,
            "break_even_occupancy": break_even, "total_invested": total_invested,
            "noi": noi,
        }


def compare_markets(purchase_price, down_pct=0.20, annual_rate=0.0725):
    """Run the same deal across all three markets side by side."""
    print(f"\n{'═'*60}")
    print(f"  MARKET COMPARISON  |  ${purchase_price:,.0f}  |  {down_pct*100:.0f}% down  |  {annual_rate*100:.2f}% rate")
    print(f"{'═'*60}")
    print(f"  {'Metric':<28} {'Indy':>8} {'LA':>10} {'Vegas':>10}")
    print(f"  {'─'*56}")

    results = {}
    for key in MARKETS:
        results[key] = analyze_deal(key, purchase_price, down_pct, annual_rate)

    indy  = results["indianapolis"]
    la    = results["los_angeles"]
    vegas = results["las_vegas"]

    def row(label, key, fmt="$"):
        if fmt == "$":
            vals = [f"${results[m][key]:>+,.0f}" for m in ["indianapolis","los_angeles","las_vegas"]]
        else:
            vals = [f"{results[m][key]:>.1f}%" for m in ["indianapolis","los_angeles","las_vegas"]]
        print(f"  {label:<28} {vals[0]:>8} {vals[1]:>10} {vals[2]:>10}")

    row("Monthly cash flow",     "monthly_cash_flow", "$")
    row("Annual cash flow",      "annual_cash_flow",  "$")
    row("Cap rate",              "cap_rate",           "%")
    row("Cash-on-cash return",   "coc_return",         "%")
    row("Gross yield",           "gross_yield",        "%")
    row("Break-even occupancy",  "break_even_occupancy", "%")
    print(f"  {'─'*56}")
    print(f"  {'Total cash invested':<28} ${indy['total_invested']:>7,.0f} ${la['total_invested']:>9,.0f} ${vegas['total_invested']:>9,.0f}")
    print(f"{'═'*60}\n")


def scenario_grid(market_key, purchase_price):
    """Show cash flow across different rate + down payment combos."""
    market = MARKETS[market_key]
    print(f"\n  SCENARIO GRID — {market['label']} — ${purchase_price:,.0f}")
    print(f"  Monthly cash flow by interest rate + down payment\n")

    rates    = [0.0625, 0.0675, 0.0725, 0.0775, 0.0825]
    down_pcts = [0.10, 0.15, 0.20, 0.25]

    header = f"  {'Rate':<8}" + "".join(f"  {int(d*100)}% down" for d in down_pcts)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for rate in rates:
        row = f"  {rate*100:.2f}%  "
        for dp in down_pcts:
            result = analyze_deal(market_key, purchase_price, dp, rate)
            cf = result["monthly_cash_flow"]
            flag = "✓" if cf >= 0 else "✗"
            row += f"  {flag}${cf:>+6,.0f}"
        print(row)
    print()


def str_vs_ltr(market_key, purchase_price, property_type="2br",
               down_pct=0.20, annual_rate=0.0725, monthly_rent=None,
               custom_adr=None, custom_occupancy=None):
    """Compare STR and LTR side by side for the same property."""
    market = MARKETS[market_key]
    str_r  = analyze_deal(market_key, purchase_price, down_pct, annual_rate,
                          rental_type="str", property_type=property_type,
                          custom_adr=custom_adr, custom_occupancy=custom_occupancy)
    ltr_r  = analyze_deal(market_key, purchase_price, down_pct, annual_rate,
                          rental_type="ltr", property_type=property_type,
                          monthly_rent=monthly_rent)

    print(f"\n{'═'*58}")
    print(f"  STR vs LTR  |  {market['label']}  |  ${purchase_price:,.0f}  |  {property_type}")
    print(f"{'═'*58}")
    print(f"  {'Metric':<28} {'STR':>12} {'LTR':>12}")
    print(f"  {'─'*54}")

    def row(label, key, fmt="$"):
        sv = str_r.get(key, 0)
        lv = ltr_r.get(key, 0)
        if fmt == "$":
            print(f"  {label:<28} ${sv:>+10,.0f} ${lv:>+10,.0f}")
        else:
            print(f"  {label:<28} {sv:>11.2f}% {lv:>11.2f}%")

    row("Monthly cash flow",   "monthly_cash_flow", "$")
    row("Annual cash flow",    "annual_cash_flow",  "$")
    row("Cap rate",            "cap_rate",           "%")
    row("Cash-on-cash return", "coc_return",         "%")
    row("Gross yield",         "gross_yield",        "%")
    print(f"  {'─'*54}")
    print(f"  {'Management effort':<28} {'High':>12} {'Low':>12}")
    print(f"  {'Regulatory risk':<28} {'Higher':>12} {'Lower':>12}")
    print(f"  {'Income stability':<28} {'Variable':>12} {'Stable':>12}")

    winner = "STR" if str_r["annual_cash_flow"] > ltr_r["annual_cash_flow"] else "LTR"
    print(f"\n  Better cash flow: {winner}")
    print(f"{'═'*58}\n")


# ── Run it ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. STR analysis
    analyze_deal("indianapolis", purchase_price=320_000, property_type="2br", rental_type="str")
    analyze_deal("las_vegas",    purchase_price=420_000, property_type="2br", rental_type="str")

    # 2. LTR analysis
    analyze_deal("indianapolis", purchase_price=320_000, property_type="2br", rental_type="ltr")
    analyze_deal("las_vegas",    purchase_price=420_000, property_type="2br", rental_type="ltr")

    # 3. STR vs LTR side by side — most useful output
    str_vs_ltr("indianapolis", purchase_price=320_000, property_type="2br")
    str_vs_ltr("las_vegas",    purchase_price=420_000, property_type="2br")

    # 4. With real numbers when you have them
    # str_vs_ltr("indianapolis", purchase_price=320_000, property_type="2br",
    #            monthly_rent=1450, custom_adr=155, custom_occupancy=0.70)

    # 5. Compare all markets STR
    compare_markets(purchase_price=400_000, down_pct=0.20, annual_rate=0.0725)

    # 6. Scenario grid
    scenario_grid("indianapolis", purchase_price=320_000)