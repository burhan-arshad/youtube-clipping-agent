"""Notifications: always stored in the app (bell icon), optionally pushed to the phone via ntfy.sh or Telegram."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import requests


class Notifier:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store

    def send(self, level: str, title: str, body: str = "") -> None:
        """level: info | ok | warn | error"""
        try:
            self.store.add_notification(level, title, body)
        except Exception:
            pass
        topic = str(self.cfg.get("notify.ntfy_topic", "") or "").strip()
        if topic:
            try:
                requests.post(f"https://ntfy.sh/{topic}", data=(body or title).encode("utf-8"), timeout=10,
                              headers={"Title": title.encode("ascii", "ignore").decode() or "featureclip",
                                       "Priority": "high" if level in ("warn", "error") else "default"})
            except Exception:
                pass
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat = str(self.cfg.get("notify.telegram_chat_id", "") or "").strip()
        if token and chat:
            try:
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=10,
                              json={"chat_id": chat, "text": f"{title}\n{body}".strip()})
            except Exception:
                pass

        to = str(self.cfg.get("notify.email_to", "") or "").strip()
        pw = os.environ.get("NOTIFY_EMAIL_PASSWORD", "").strip()
        if to and pw:
            try:        # Gmail: use an "App password", not your normal password
                msg = EmailMessage()
                msg["From"], msg["To"], msg["Subject"] = to, to, f"[featureclip] {title}"
                msg.set_content(body or title)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
                    s.login(to, pw)
                    s.send_message(msg)
            except Exception:
                pass
