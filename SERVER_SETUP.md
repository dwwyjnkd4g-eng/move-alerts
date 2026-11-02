# Moving the bot to an always-on server

About 45 minutes, once. After this:

- squawk headlines arrive within seconds
- every job runs on exact timings
- the Telegram commands work

## 1. Create the server (10 min)

1. Sign up at **digitalocean.com** and add a card. A basic server costs about $6 a month.
2. Click **Create → Droplets**.
3. Choose these settings:
   - **Region:** London.
   - **Image:** Ubuntu 24.04.
   - **Size:** Basic, Regular, the $6 option (1 GB RAM).
   - **Authentication:** Password. Choose a strong one and save it.
   - **Hostname:** `move-alerts`.
4. Click **Create Droplet** and wait about a minute.
5. Click the droplet, then **Access → Launch Droplet Console**. A terminal opens in your browser.

## 2. Install the bot (5 min)

In the console, paste this command (one line), using your repo URL, and press Enter:

```
curl -sSL https://raw.githubusercontent.com/dwwyjnkd4g-eng/move-alerts/main/setup.sh | bash -s -- https://github.com/dwwyjnkd4g-eng/move-alerts.git
```

## 3. Get your Telegram API ID (3 min)

1. On your phone or computer, go to **my.telegram.org** and log in with your phone number. The code arrives in Telegram.
2. Click **API development tools**.
3. Fill in an app title (`move-alerts`) and a short name (`movealerts`). Choose platform **Other**, then click **Create application**.
4. Copy the **api_id** (a number) and the **api_hash** (a long string).

## 4. Add your keys (5 min)

In the console:

```
nano ~/move-alerts/.env
```

Fill in each line after the `=`. Use the same values as your GitHub secrets, plus `TG_API_ID` and `TG_API_HASH`. To save, press **Ctrl+O**, then **Enter**, then **Ctrl+X**.

## 5. Log the listener in to Telegram (2 min, once)

```
cd ~/move-alerts && set -a && . ./.env && set +a && .venv/bin/python listener.py --login
```

1. Enter your phone number in international format, for example `+447...`.
2. Enter the code Telegram sends you.
3. If you have a Telegram 2FA password, enter that too.
4. You should see "Logged in as …".

## 6. Add your holdings (1 min)

```
.venv/bin/python holdings.py --seed "TICKER:Name,TICKER2:Name2|Alias,LSE.L"
```

The names after the colons let the squawk listener spot news about your companies.

Quantities and costs can be added later from Telegram, for example `/add AAPL 10 @ 225`.

## 7. Start everything

```
sudo systemctl enable --now move-alerts-runner move-alerts-listener move-alerts-bot
```

Check that all three services say `active (running)`:

```
systemctl status move-alerts-runner move-alerts-listener move-alerts-bot --no-pager | grep -E "●|Active"
```

## 8. Turn off the GitHub schedules

This stops you getting duplicate alerts. In your repo, go to **Actions**. For each of **US move alerts**, **Macro alerts** and **Daily brief**, click it, then **⋯ → Disable workflow**.

## 9. Test from Telegram

Send your bot these commands:

- `/help`
- `/levels`
- `/holdings`
- `/why NVDA`
- `/brief`

## Useful commands later

| What | Command |
|---|---|
| Update after I change code | `bash ~/move-alerts/update.sh` |
| Live log of the squawk listener | `journalctl -u move-alerts-listener -f` |
| Live log of the scheduled jobs | `journalctl -u move-alerts-runner -f` |
| Restart everything | `sudo systemctl restart move-alerts-runner move-alerts-listener move-alerts-bot` |

## Security notes

- The file `~/move-alerts/squawk.session` lets this server read Telegram as you. Keep the server password safe.
- To revoke the server's access, open Telegram and go to **Settings → Devices**.
- Your holdings and keys stay on the server and never go to GitHub (`.gitignore` covers them).
