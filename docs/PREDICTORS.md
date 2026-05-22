# Predictor catalog

Free data only. Mark conventions:
- **[Z]** zip-native · **[T]** tract-native (zip-mapped via crosswalk) · **[C]** county/metro only (broadcast)
- **★** well-documented signal · **🧪** speculative, worth testing
- **🎯** especially well-suited to zip-level work

---

## 1. Money & rates (timing)
| Predictor                              | Level | Conf | Source       |
|----------------------------------------|-------|------|--------------|
| 30Y mortgage rate + Δ                  | C     | ★    | FRED MORTGAGE30US |
| 10Y TIPS / real rates                  | C     | ★    | FRED DFII10  |
| M2 YoY                                 | C     | ★    | FRED M2SL    |
| MBA mortgage application index (hdr.)  | C     | ★    | FRED         |
| Housing affordability index            | C     | ★    | NAR via FRED |
| Cash-out refi share                    | C     | 🧪   | Freddie Mac  |
| 30-day mortgage delinquencies          | C     | 🧪   | NY Fed CCP   |

## 2. Demographics & migration 🎯
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| USPS COA / vacancy aggregates          | Z     | ★ 🎯 | HUD-USPS quarterly       |
| County-to-county AGI migration         | C     | ★    | IRS SOI                  |
| ACS demographics (age, income, edu)    | Z     | ★    | Census ACS (5-yr)        |
| University enrollment                  | C     | ★    | IPEDS                    |
| Public-school enrollment by district   | T     | ★    | NCES                     |
| Boomer mortality share (age 75+)       | Z     | 🧪   | Census ACS               |
| Birth rate (lagged 30y)                | C     | 🧪   | CDC NCHS                 |
| LinkedIn workforce migration           | C     | 🧪   | LinkedIn Workforce Reports |

## 3. Jobs & income 🎯
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| Wages + employment by industry         | C     | ★    | BLS QCEW                 |
| Applicant income distribution          | T     | ★ 🎯 | HMDA                     |
| Federal contract dollars               | Z     | 🧪   | USAspending.gov          |
| Local area unemployment                | C     | ★    | BLS LAUS                 |
| Hospital bed count growth              | C     | 🧪   | AHA / CMS POS            |
| Remote-work intensity                  | C     | 🧪   | BLS ATUS + Kastle        |

## 4. Supply & construction
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| Building permits                       | C     | ★    | Census BPS               |
| Months-of-supply, DOM, price-cut %     | Z     | ★ 🎯 | Realtor.com research     |
| Sale-to-list, inventory                | Z     | ★ 🎯 | Redfin Data Center       |
| Wharton land-use regulation index      | C     | ★    | Wharton WRLURI 2018      |
| Satellite-detected new construction    | Z     | 🧪   | NASA / open imagery      |
| Lumber + steel price index             | C     | ★    | FRED                     |

## 5. Amenity / "Whole Foods class" 🎯
All scrape-able from Foursquare Open Places (free, ~100M POIs).
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| WF / TJ / Wegmans / Costco openings    | Z     | ★ 🎯 | Foursquare OS Places     |
| Erewhon / Sweetgreen / Equinox         | Z     | 🧪   | Foursquare OS Places     |
| Third-wave coffee density              | Z     | 🧪   | Foursquare OS Places     |
| Brewery openings                       | Z     | 🧪   | Brewers Association      |
| Pilates / boutique fitness density     | Z     | 🧪   | Foursquare OS Places     |
| Independent bookstores                 | Z     | 🧪   | Foursquare OS Places     |
| Pet groomer / daycare density          | Z     | 🧪   | Foursquare OS Places     |
| Pickleball courts                      | Z     | 🧪   | OSM                      |
| Drag brunch listings                   | Z     | 🧪   | OSM / Eventbrite scrape  |
| Michelin / NYT 50 Best                 | Z     | 🧪   | Michelin / NYT           |

## 6. Lifestyle & search / behavior 🎯
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| Google Trends: housing queries         | C     | ★    | Google Trends            |
| Walk / Bike / Transit score            | Z     | ★    | Walk Score (free at zip) |
| Strava heatmap density                 | Z     | 🧪   | Strava public            |
| Wikipedia page-view trend              | Z     | 🧪   | Wikimedia stats          |
| Reddit city-sub growth + sentiment     | C     | 🧪   | PRAW                     |
| Yelp opening:closing ratio             | Z     | 🧪   | Yelp Fusion (free tier)  |
| TikTok geotagged volume                | Z     | 🧪   | TikTok API               |

## 7. Climate & insurance 🎯 (underweighted sleeper)
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| First Street flood/fire/heat score     | Z     | ★ 🎯 | CEJST + state dashboards |
| **Insurance premium Δ + non-renewals** | C     | ★ 🎯 | State insurance dept filings |
| NAIC aggregated premium data           | C     | ★    | NAIC                     |
| FEMA disaster declarations             | Z     | ★    | FEMA OpenFEMA            |
| NOAA heat-day projections              | Z     | ★    | NOAA                     |
| Reinsurance pricing                    | C     | 🧪   | Guy Carpenter index      |

## 8. Infrastructure & policy
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| New transit stations                   | Z     | ★    | FTA NTD + city open data |
| Airport nonstop-route additions        | C     | ★    | BTS T-100                |
| Federal IIJA / IRA spend               | Z     | 🧪   | USAspending.gov          |
| Fiber / gigabit availability           | T     | ★    | FCC Form 477             |
| EV charger density                     | Z     | 🧪   | DOE AFDC                 |
| Solar permit issuance                  | Z     | 🧪   | City open data           |
| State + local tax burden Δ             | C     | ★    | Tax Foundation / state revenue |
| ADU / upzoning legislation             | C     | 🧪   | YIMBY trackers           |

## 9. Capital flows
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| Institutional buyer share              | T     | ★ 🎯 | HMDA (derived)           |
| Cash sale %                            | C     | ★    | NAR                      |
| Foreign cash buyer activity            | C     | 🧪   | FinCEN GTO (limited)     |
| iBuyer buying patterns                 | Z     | 🧪   | Opendoor / Offerpad pub. |

## 10. Genuinely weird stuff 🧪
| Predictor                              | Level | Conf | Source                  |
|----------------------------------------|-------|------|--------------------------|
| NOAA VIIRS nighttime lights Δ          | Z     | 🧪 🎯 | NASA Earth Data          |
| Satellite swimming-pool count          | Z     | 🧪   | Sentinel-2 + ML          |
| Parking-lot fullness                   | Z     | 🧪   | Orbital Insight free tier|
| Pet license registrations              | Z     | 🧪   | City open data           |
| Private school waitlist length         | Z     | 🧪   | Scrapeable               |
| 311 complaint density + category mix   | Z     | 🧪   | City open data           |
| Cell-tower daytime/nighttime pop ratio | Z     | 🧪   | (paid; free proxies exist) |
| FBI NIBRS crime trends                 | C     | ★    | FBI NIBRS                |

---

## Phase 1 starter pack (15 predictors)

Highest signal-to-effort to validate the approach before expanding:

1. 30Y mortgage rate Δ (FRED)
2. HUD-USPS net inflow per zip (HUD)
3. IRS SOI net AGI inflow (IRS)
4. ACS age 28–38 share (Census)
5. HMDA origination volume + applicant income (CFPB)
6. Realtor.com months-of-supply, DOM, price-cut rate (Realtor)
7. Census BPS building permits (Census)
8. BLS QCEW wage growth (BLS)
9. Foursquare Whole Foods / Trader Joe's count Δ
10. Foursquare third-wave coffee + brewery count Δ
11. First Street flood + fire score (CEJST)
12. FEMA disaster declaration count, 10-yr (FEMA)
13. NOAA VIIRS nightlight Δ (NASA)
14. FCC broadband availability (FCC)
15. HMDA institutional-buyer share (derived)
