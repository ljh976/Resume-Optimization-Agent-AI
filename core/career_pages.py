# Static listing of company career pages for display in the UI.
# Two groups: FORTUNE_30 (representative Fortune 30 companies) and BIG_TECH.
# Each entry is a dict: {"name": <display name>, "url": <careers page url>}.

FORTUNE_30 = [
    {"name": "Walmart", "url": "https://careers.walmart.com/"},
    {"name": "Amazon", "url": "https://www.amazon.jobs/"},
    {"name": "Apple", "url": "https://www.apple.com/careers/"},
    {"name": "CVS Health", "url": "https://jobs.cvshealth.com/"},
    {"name": "Berkshire Hathaway", "url": "https://www.berkshirehathaway.com/"},
    {"name": "UnitedHealth Group", "url": "https://www.unitedhealthgroup.com/careers.html"},
    {"name": "McKesson", "url": "https://www.mckesson.com/careers/"},
    {"name": "AT&T", "url": "https://www.att.jobs/"},
    {"name": "AmerisourceBergen", "url": "https://www.amerisourcebergen.com/careers"},
    {"name": "Chevron", "url": "https://careers.chevron.com/"},
    {"name": "Ford Motor", "url": "https://corporate.ford.com/careers.html"},
    {"name": "Cigna", "url": "https://jobs.cigna.com/"},
    {"name": "Costco", "url": "https://www.costco.com/jobs.html"},
    {"name": "Honda Motor", "url": "https://global.honda/career.html"},
    {"name": "Cardinal Health", "url": "https://www.cardinalhealth.com/en/careers.html"},
    {"name": "Walgreens Boots Alliance", "url": "https://jobs.walgreens.com/"},
    {"name": "Toyota Motor", "url": "https://www.toyota-global.com/jobs/"},
    {"name": "Volkswagen", "url": "https://www.vw-careers.com/"},
    {"name": "Exxon Mobil", "url": "https://jobs.exxonmobil.com/"},
    {"name": "The Home Depot", "url": "https://careers.homedepot.com/"},
    {"name": "Procter & Gamble", "url": "https://www.pgcareers.com/"},
    {"name": "Target", "url": "https://corporate.target.com/careers"},
    {"name": "CVS Health (retained)", "url": "https://jobs.cvshealth.com/"},
    {"name": "Nestle", "url": "https://www.nestle.com/jobs"},
    {"name": "Johnson & Johnson", "url": "https://www.careers.jnj.com/"},
    {"name": "Kroger", "url": "https://jobs.kroger.com/"},
    {"name": "Verizon", "url": "https://www.verizon.com/about/careers/"},
    {"name": "Comcast", "url": "https://jobs.comcast.com/"},
    {"name": "Disney", "url": "https://jobs.disneycareers.com/"},
    {"name": "Pfizer", "url": "https://www.pfizer.com/careers"},
]

BIG_TECH = [
    {"name": "Google / Alphabet", "url": "https://careers.google.com/"},
    {"name": "Microsoft", "url": "https://careers.microsoft.com/"},
    {"name": "Apple", "url": "https://www.apple.com/careers/"},
    {"name": "Amazon", "url": "https://www.amazon.jobs/"},
    {"name": "Meta (Facebook)", "url": "https://www.metacareers.com/"},
    {"name": "Netflix", "url": "https://jobs.netflix.com/"},
    {"name": "Tesla", "url": "https://www.tesla.com/careers"},
    {"name": "NVIDIA", "url": "https://www.nvidia.com/en-us/about-nvidia/careers/"},
    {"name": "Salesforce", "url": "https://www.salesforce.com/company/careers/"},
    {"name": "Stripe", "url": "https://stripe.com/jobs"},
    {"name": "Adobe", "url": "https://www.adobe.com/careers.html"},
    {"name": "Oracle", "url": "https://www.oracle.com/corporate/careers/"},
    {"name": "Intel", "url": "https://jobs.intel.com/"},
    {"name": "Cisco", "url": "https://jobs.cisco.com/"},
    {"name": "Shopify", "url": "https://www.shopify.com/careers"},
    {"name": "Uber", "url": "https://www.uber.com/global/en/careers/"},
    {"name": "Airbnb", "url": "https://careers.airbnb.com/"},
    {"name": "Snap Inc.", "url": "https://www.snap.com/en-US/jobs"},
    {"name": "Palantir", "url": "https://www.palantir.com/careers/"},
    {"name": "Qualcomm", "url": "https://www.qualcomm.com/company/careers"},
    {"name": "AMD", "url": "https://www.amd.com/en/careers"},
    {"name": "Dropbox", "url": "https://www.dropbox.com/jobs"},
    {"name": "Atlassian", "url": "https://www.atlassian.com/company/careers"},
    {"name": "GitHub", "url": "https://github.com/about/careers"},
]

# Companies that are commonly a good fit for experienced engineers/product roles
SUGGESTED = [
    {"name": "Databricks", "url": "https://databricks.com/company/careers"},
    {"name": "Snowflake", "url": "https://www.snowflake.com/careers/"},
    {"name": "Stripe", "url": "https://stripe.com/jobs"},
    {"name": "Square / Block", "url": "https://block.com/careers"},
    {"name": "Shopify", "url": "https://www.shopify.com/careers"},
    {"name": "Reddit", "url": "https://www.redditinc.com/careers"},
    {"name": "Lyft", "url": "https://www.lyft.com/careers"},
    {"name": "Pinterest", "url": "https://careers.pinterest.com/"},
    {"name": "Coinbase", "url": "https://www.coinbase.com/careers"},
    {"name": "Okta", "url": "https://www.okta.com/company/careers/"},
    {"name": "PagerDuty", "url": "https://www.pagerduty.com/careers/"},
    {"name": "Twilio", "url": "https://www.twilio.com/company/jobs"},
    {"name": "Datadog", "url": "https://www.datadoghq.com/careers/"},
    {"name": "HashiCorp", "url": "https://www.hashicorp.com/careers"},
    {"name": "Confluent", "url": "https://www.confluent.io/careers/"},
]

# Combined helper and categorized map
ALL_COMPANIES = FORTUNE_30 + BIG_TECH + SUGGESTED
CATEGORIZED = {
    "Fortune & Large Employers": FORTUNE_30,
    "Big Tech & Platform Companies": BIG_TECH,
    "Suggested (Good Fit)": SUGGESTED,
}
