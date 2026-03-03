---
id: current-datetime
title: Current Datetime Grounding
description: Anchor time-sensitive reasoning with exact current UTC date and time.
tags: [time, grounding, recency]
---
Use current UTC datetime when user requests include terms such as "today", "latest",
"current", "now", or deadlines/schedules that depend on real-time context.

If date/time grounding is needed before external lookups, resolve current time first and
use it as context for subsequent planning and answering.
