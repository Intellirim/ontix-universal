# Lemonsqueezy Setup Guide — ONTIX Industry Packs

## 1. Store Setup (One-time)

1. Go to https://lemonsqueezy.com and sign in
2. Store name: **Intellirim** (or your preferred name)
3. Complete Stripe Connect setup for payouts

## 2. Create 7 Products

Create each as **Digital Download** product:

| # | Product Name | Price | ZIP File to Upload |
|---|---|---|---|
| 1 | ONTIX Beauty & Skincare Pack | $79 | `dist/ontix-beauty-pack.zip` |
| 2 | ONTIX Food & Beverage Pack | $79 | `dist/ontix-fnb-pack.zip` |
| 3 | ONTIX Fashion & Apparel Pack | $79 | `dist/ontix-fashion-pack.zip` |
| 4 | ONTIX Tech / SaaS Pack | $79 | `dist/ontix-tech-saas-pack.zip` |
| 5 | ONTIX Fitness & Wellness Pack | $79 | `dist/ontix-fitness-pack.zip` |
| 6 | ONTIX Entertainment Pack | $79 | `dist/ontix-entertainment-pack.zip` |
| 7 | ONTIX All 6 Packs Bundle | $249 | `dist/ontix-all-packs-bundle.zip` |

### For each product:
1. **Products** → **New Product**
2. Name: (from table above)
3. Price: (from table above)
4. **Files**: Upload the corresponding ZIP
5. **Fulfillment**: "Send file after payment" (automatic delivery)
6. Description (copy-paste for each):

**Individual Pack:**
> Domain-optimized extraction prompt, brand config, and 10-12 production Cypher queries for ONTIX Universal. Drop into your packs/ directory and restart.

**Bundle:**
> All 6 Industry Packs (Beauty, F&B, Fashion, Tech/SaaS, Fitness, Entertainment) at 47% discount. Includes all future pack updates.

## 3. Get Checkout URLs

After creating each product:
1. Go to **Products** → click the product
2. Click **Share** → **Checkout link**
3. Copy the URL (format: `https://intellirim.lemonsqueezy.com/buy/VARIANT_ID`)

## 4. Paste URLs into Landing Page

Open `docs/index.html` and `index.html`, find the `CHECKOUT_LINKS` object (~line 798):

```javascript
const CHECKOUT_LINKS = {
  'beauty':        'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'fnb':           'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'fashion':       'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'tech-saas':     'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'fitness':       'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'entertainment': 'https://intellirim.lemonsqueezy.com/buy/XXXXX',
  'bundle':        'https://intellirim.lemonsqueezy.com/buy/XXXXX',
};
```

Replace each `XXXXX` with the actual variant ID from Lemonsqueezy.

## 5. Test

1. Open the landing page locally
2. Click "Buy Pack" → should open Lemonsqueezy checkout
3. Use test mode to verify purchase flow
4. Confirm ZIP is delivered via email after purchase

## 6. Optional: Lemonsqueezy Overlay Checkout

For a smoother UX (checkout opens as overlay instead of redirect), add this before `</head>`:

```html
<script src="https://assets.lemonsqueezy.com/lemon.js" defer></script>
```

Then change checkout links to use `data-lemonsqueezy` attribute. See Lemonsqueezy docs for details.
