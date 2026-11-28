import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ── Linux / Streamlit Cloud SQLite3 Compatibility Shim for ChromaDB ──────────
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

import requests
import streamlit as st
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_secret(key: str, default: str = "") -> Tuple[str, bool]:
    """
    Retrieve setting from Streamlit secrets first (`st.secrets`), falling back to `os.environ`.
    Returns (value, is_from_st_secrets).
    """
    try:
        if key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip() != "":
                return str(val).strip(), True
    except Exception:
        pass
    env_val = os.environ.get(key, "").strip()
    if env_val:
        return env_val, False
    return default, False


# ── App Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="دستیار هوشمند پارسین طب (OpenAI)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Persian RTL CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    html, body, [class*="css"], .stMarkdown, .stChatInput, .stChatMessage, p, h1, h2, h3, h4, span, div, label {
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .stChatMessage {
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    .product-badge-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 8px;
        display: inline-block;
        margin-left: 8px;
        transition: transform 0.15s ease;
        text-decoration: none !important;
    }
    .product-badge-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.15);
    }
    .product-name {
        color: #065f46;
        font-weight: 700;
        font-size: 13px;
    }
    .product-price {
        color: #047857;
        font-weight: 600;
        font-size: 12px;
        margin-top: 2px;
    }

    [data-testid="stSidebar"] {
        direction: rtl !important;
        text-align: right !important;
    }

    .app-header {
        background: linear-gradient(90deg, #059669 0%, #0d9488 100%);
        padding: 18px 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
    }
    .app-header h1 {
        color: white !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }
    .app-header p {
        color: #e6fffa !important;
        font-size: 13px !important;
        margin: 4px 0 0 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths & Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
FALLBACK_CATALOG_PATH = BASE_DIR / "data" / "products_catalog.json"
CONTEXTUAL_TRIGGERS = {
    'این', 'اون', 'آن', 'همین', 'کدوم', 'کدام', 'قیمتش', 'چنده',
    'چند', 'قیمت', 'گارانتی', 'هزینه', 'موجوده', 'موجود', 'داره',
    'ارسال', 'چطوره', 'مشخصاتش', 'کیفیتش', 'رنگ', 'سایز', 'تخفیف'
}
EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def to_persian_digits(number_str: Any) -> str:
    """Convert western digits to Persian characters."""
    fa_digits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
    return re.sub(r'\d', lambda m: fa_digits[int(m.group(0))], str(number_str))


def normalize_persian(text: str) -> str:
    """Normalize Arabic/Persian characters and whitespace for reliable matching."""
    if not text:
        return ""
    s = str(text)
    s = s.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    fa_to_en = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    s = s.translate(fa_to_en)
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_bbcode(text: str) -> str:
    """Remove custom color BBCode tags ([green], [blue], [red], etc.) from product descriptions."""
    if not text:
        return ""
    return re.sub(r"\[/?[a-zA-Z]+\]", "", str(text)).strip()


def compute_product_pricing(product: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Return (final_price, original_price, discount_percent) matching the React frontend logic.
    """
    price_raw = product.get("price", 0)
    try:
        original_price = int(float(price_raw or 0))
    except (ValueError, TypeError):
        original_price = 0

    discount_obj = product.get("current_discount")
    discount_pct = 0
    if isinstance(discount_obj, dict) and discount_obj.get("is_active"):
        try:
            discount_pct = int(float(discount_obj.get("discount_percentage") or 0))
        except (ValueError, TypeError):
            discount_pct = 0

    if discount_pct > 0 and original_price > 0:
        final_price = int(round(original_price * (1.0 - discount_pct / 100.0)))
    else:
        final_price = original_price

    return final_price, original_price, discount_pct


def compute_in_stock(product: Dict[str, Any]) -> Tuple[bool, int]:
    """Return (in_stock, stock_count) accurately from product payload."""
    if "stock" in product:
        try:
            cnt = int(product.get("stock") or 0)
            return cnt > 0, max(0, cnt)
        except (ValueError, TypeError):
            in_s = bool(product.get("stock"))
            return in_s, (1 if in_s else 0)
    in_s = bool(product.get("in_stock", True))
    return in_s, (1 if in_s else 0)


# ── Standalone OpenAI ChromaDB Setup ─────────────────────────────────────────
class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB embedding function connecting to OpenAI Embeddings API without silent zero-vector poisoning."""

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.openai.com/v1',
        model: str = 'text-embedding-3-small'
    ):
        self.api_key = (api_key or '').strip()
        self.base_url = (base_url or 'https://api.openai.com/v1').strip().rstrip('/')
        self.model = (model or 'text-embedding-3-small').strip()

    def name(self) -> str:
        return f"openai_{self.model}"

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY تنظیم نشده است.")

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(input)},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"OpenAI Embeddings HTTP {resp.status_code}: {resp.text[:250]}")

        data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        embeddings = [item.get("embedding", []) for item in items if item.get("embedding")]
        if not embeddings or len(embeddings) != len(input):
            raise RuntimeError("پاسخ نامعتبر از سرویس Embeddings دریافت شد.")
        return embeddings


@st.cache_resource(show_spinner=False)
def get_chroma_collection_openai(chroma_dir: str, api_key: str, base_url: str, embed_model: str, collection_name: str):
    """Cached OpenAI ChromaDB collection loader (falls back to EphemeralClient if disk is read-only)."""
    ef = OpenAIEmbeddingFunction(
        api_key=api_key,
        base_url=base_url,
        model=embed_model,
    )
    # Bump collection suffix to v2 so any previously poisoned zero-vector cache is cleanly bypassed
    safe_collection_name = f"{collection_name}_v2"
    try:
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_or_create_collection(name=safe_collection_name, embedding_function=ef)
    except Exception as e:
        logger.warning(f"[ChromaDB] PersistentClient failed ({e}), falling back to EphemeralClient.")
        try:
            client = chromadb.EphemeralClient()
            return client.get_or_create_collection(name=safe_collection_name, embedding_function=ef)
        except Exception as inner_err:
            logger.error(f"[ChromaDB] EphemeralClient also failed: {inner_err}")
            return None


def format_product_dict_document(product: Dict[str, Any]) -> str:
    """Format a product JSON object into structured Persian text for RAG retrieval."""
    in_stock, stock_count = compute_in_stock(product)
    status_text = f"موجود در انبار ({stock_count} عدد)" if in_stock else "ناموجود"

    final_price, original_price, discount_pct = compute_product_pricing(product)
    if final_price > 0 and discount_pct > 0:
        price_line = f"{final_price:,} تومان (با {discount_pct}% تخفیف؛ قیمت قبل: {original_price:,} تومان)"
    elif final_price > 0:
        price_line = f"{final_price:,} تومان"
    else:
        price_line = "تماس بگیرید"

    colors = product.get("colors") or []
    sizes = [s for s in (product.get("size") or []) if str(s).upper() != "NA"]

    doc_lines = [
        f"شناسه کالا: {product.get('id', '')}",
        f"نام محصول: {product.get('name', 'نامشخص')}",
        f"دسته‌بندی: {product.get('category') or 'عمومی'}",
        f"برند: {product.get('brand') or 'مشخص نشده'}",
        f"قیمت نهایی فروش: {price_line}",
        f"وضعیت موجودی: {status_text}",
    ]
    if colors:
        doc_lines.append(f"رنگ‌های موجود: {'، '.join(str(c) for c in colors)}")
    if sizes:
        doc_lines.append(f"سایزبندی: {'، '.join(str(s) for s in sizes)}")

    desc = clean_bbcode(product.get("description") or "")
    if desc:
        doc_lines.append(f"توضیحات و مشخصات تخصصی:\n{desc}")

    return "\n".join(doc_lines)


def fetch_products_from_api(products_api_url: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetch all products from the Django REST API endpoint.
    Supports both DRF paginated responses (`{"results": [...], "next": ...}`) and direct JSON lists.
    """
    if not products_api_url or not products_api_url.strip():
        return [], "آدرس API محصولات (`PRODUCTS_API_URL`) تنظیم نشده است."

    url = products_api_url.strip()
    if "?" not in url:
        url = f"{url}?page_size=100"

    all_products: List[Dict[str, Any]] = []
    visited_urls = set()

    try:
        while url and url not in visited_urls and len(visited_urls) < 20:
            visited_urls.add(url)
            resp = requests.get(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; ParsinTebRAG/1.0)",
                },
                timeout=8,
            )
            if not resp.ok:
                return [], f"خطای HTTP {resp.status_code} از سرور محصولات ({url})"

            payload = resp.json()
            if isinstance(payload, list):
                all_products.extend([item for item in payload if isinstance(item, dict)])
                break
            elif isinstance(payload, dict):
                results = payload.get("results", payload.get("data", []))
                if isinstance(results, list):
                    all_products.extend([item for item in results if isinstance(item, dict)])
                url = payload.get("next")
            else:
                break

        return all_products, ""
    except requests.exceptions.ConnectTimeout as e:
        logger.warning(f"[Products API] Connection timed out for {products_api_url}: {e}")
        return [], f"اتصال مستقیم به {products_api_url} برقرار نشد (Timeout)."
    except requests.exceptions.Timeout as e:
        logger.warning(f"[Products API] Request timed out for {products_api_url}: {e}")
        return [], f"دریافت پاسخ از API محصولات بیش از زمان مجاز طول کشید: {e}"
    except Exception as e:
        logger.warning(f"[Products API] Failed to fetch from {products_api_url}: {e}")
        return [], str(e)


def load_fallback_catalog(catalog_path: Path = FALLBACK_CATALOG_PATH) -> List[Dict[str, Any]]:
    """Load bundled fallback product catalog JSON when remote API is unreachable from cloud."""
    try:
        if catalog_path.exists():
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                results = data.get("results", data.get("data", []))
                if isinstance(results, list):
                    return [item for item in results if isinstance(item, dict)]
    except Exception as e:
        logger.warning(f"[Fallback Catalog] Failed to read {catalog_path}: {e}")
    return []


@st.cache_data(ttl=600, show_spinner=False)
def get_active_catalog(products_api_url: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Return `(products_list, used_fallback)` cached for 10 minutes so we always have
    the full product list in memory for 100% accurate catalog overview + hybrid search.
    """
    products, err = fetch_products_from_api(products_api_url)
    if not err and products:
        return products, False
    fallback = load_fallback_catalog()
    if fallback:
        return fallback, True
    return [], False


def build_product_metadata(prod: Dict[str, Any], idx: int, frontend_url: str = "") -> Dict[str, Any]:
    prod_id = prod.get("id", prod.get("product_id", idx + 1))
    try:
        prod_id_int = int(prod_id)
    except (ValueError, TypeError):
        prod_id_int = idx + 1

    final_price, original_price, discount_pct = compute_product_pricing(prod)
    in_stock, stock_count = compute_in_stock(prod)

    clean_frontend = (frontend_url or "").strip().rstrip("/")
    rel_url = f"/products/{prod_id_int}"
    full_url = f"{clean_frontend}{rel_url}" if clean_frontend else str(prod.get("url") or rel_url)

    return {
        "product_id": prod_id_int,
        "name": str(prod.get("name") or "محصول بدون نام"),
        "category": str(prod.get("category") or ""),
        "brand": str(prod.get("brand") or ""),
        "price": final_price,
        "original_price": original_price,
        "discount_percent": discount_pct,
        "in_stock": in_stock,
        "stock": stock_count,
        "url": full_url,
    }


def sync_products_to_chroma(
    collection,
    products: List[Dict[str, Any]],
    frontend_url: str = "",
) -> Tuple[int, str]:
    """Batch-upsert products into ChromaDB with real OpenAI embeddings."""
    if collection is None:
        return 0, "پایگاه برداری ChromaDB آماده نیست."
    if not products:
        return 0, "لیست محصولات خالی است."

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, prod in enumerate(products):
        meta = build_product_metadata(prod, idx, frontend_url)
        ids.append(str(meta["product_id"]))
        documents.append(format_product_dict_document(prod))
        metadatas.append(meta)

    try:
        batch_size = 50
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        return len(ids), ""
    except Exception as e:
        logger.error(f"[ChromaDB Sync] Upsert failed: {e}")
        return 0, f"خطا در بردارسازی محصولات با OpenAI: {e}"


def build_catalog_overview_index(products: List[Dict[str, Any]]) -> str:
    """
    Build a concise 1-line-per-product inventory summary of ALL store products.
    With ~25-80 products, this costs very few tokens and guarantees the LLM has
    complete visibility over every item, category, brand, exact price, and stock!
    """
    if not products:
        return "اطلاعات فهرست کلی کالاها در دسترس نیست."

    lines = []
    for idx, p in enumerate(products, 1):
        final_price, orig_price, disc = compute_product_pricing(p)
        in_stock, stock_cnt = compute_in_stock(p)
        price_txt = f"{final_price:,} تومان" if final_price > 0 else "تماس بگیرید"
        if disc > 0:
            price_txt += f" ({disc}% تخفیف از {orig_price:,})"
        stock_txt = f"موجود ({stock_cnt} عدد)" if in_stock else "ناموجود"
        name = p.get("name", "بدون نام")
        cat = p.get("category") or "عمومی"
        brand = p.get("brand") or "-"
        lines.append(f"{idx}. {name} | دسته: {cat} | برند: {brand} | قیمت: {price_txt} | وضعیت: {stock_txt}")
    return "\n".join(lines)


def hybrid_search_products(
    collection,
    all_products: List[Dict[str, Any]],
    query_text: str,
    n_results: int = 5,
    frontend_url: str = "",
) -> List[Dict[str, Any]]:
    """
    Combine Persian lexical/keyword matching across all products with ChromaDB semantic vector search
    so exact product/category/brand queries NEVER miss any relevant items.
    """
    norm_q = normalize_persian(query_text)
    q_tokens = [t for t in re.findall(r"\w+", norm_q) if len(t) >= 2 and t not in CONTEXTUAL_TRIGGERS]

    # 1. Lexical score over full catalog
    lexical_hits: List[Tuple[float, Dict[str, Any]]] = []
    prod_by_id: Dict[int, Dict[str, Any]] = {}

    for idx, prod in enumerate(all_products):
        meta = build_product_metadata(prod, idx, frontend_url)
        pid = meta["product_id"]
        prod_by_id[pid] = {
            "document": format_product_dict_document(prod),
            "metadata": meta,
        }

        name_norm = normalize_persian(prod.get("name", ""))
        cat_norm = normalize_persian(prod.get("category", ""))
        brand_norm = normalize_persian(prod.get("brand", ""))
        desc_norm = normalize_persian(clean_bbcode(prod.get("description", "")))

        score = 0.0
        if norm_q and norm_q in name_norm:
            score += 8.0
        for tok in q_tokens:
            if tok in name_norm:
                score += 4.0
            if tok in cat_norm:
                score += 3.5
            if tok in brand_norm:
                score += 3.0
            if tok in desc_norm:
                score += 1.0
        if score > 0:
            lexical_hits.append((score, prod_by_id[pid]))

    lexical_hits.sort(key=lambda x: x[0], reverse=True)

    # 2. Vector semantic search from ChromaDB
    vector_hits: List[Dict[str, Any]] = []
    if collection is not None:
        try:
            total_in_col = collection.count()
            k = min(max(n_results, 5), total_in_col) if total_in_col > 0 else 0
            if k > 0:
                results = collection.query(query_texts=[query_text], n_results=k)
                if results and results.get("documents") and len(results["documents"]) > 0:
                    docs = results["documents"][0]
                    metas = results.get("metadatas", [[]])[0]
                    for doc, meta in zip(docs, metas):
                        pid = int(meta.get("product_id", 0)) if meta else 0
                        if pid in prod_by_id:
                            vector_hits.append(prod_by_id[pid])
                        else:
                            vector_hits.append({"document": doc, "metadata": meta})
        except Exception as e:
            logger.warning(f"[Hybrid Search] Vector query error: {e}")

    # 3. Merge deduplicated: strong lexical matches first, then semantic vector matches
    merged: List[Dict[str, Any]] = []
    seen_ids = set()

    for score, item in lexical_hits:
        if score >= 3.0:
            pid = item["metadata"].get("product_id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                merged.append(item)
            if len(merged) >= n_results:
                break

    for item in vector_hits:
        if len(merged) >= n_results:
            break
        pid = item["metadata"].get("product_id")
        if pid not in seen_ids:
            seen_ids.add(pid)
            merged.append(item)

    # Fill remaining slots with lower-score lexical hits if needed
    for _, item in lexical_hits:
        if len(merged) >= n_results:
            break
        pid = item["metadata"].get("product_id")
        if pid not in seen_ids:
            seen_ids.add(pid)
            merged.append(item)

    return merged[:n_results]


# ── LangChain OpenAI Helpers ─────────────────────────────────────────────────
def get_llm(api_key: str, base_url: str, model_name: str, temperature: float, max_tokens: int = 600) -> ChatOpenAI:
    kwargs = {
        "model": model_name,
        "api_key": api_key or "missing-openai-api-key",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip().rstrip('/')
    return ChatOpenAI(**kwargs)


def contextualize_query(user_message: str, chat_history: List[BaseMessage], llm_fast: ChatOpenAI) -> str:
    """Rewrite ambiguous follow-up questions using past conversational turns."""
    if not chat_history:
        return user_message

    words = set(re.findall(r'\w+', user_message))
    is_ambiguous = len(words) <= 4 or bool(words & CONTEXTUAL_TRIGGERS)
    if not is_ambiguous:
        return user_message

    try:
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "با توجه به سابقه پیام‌ها، پرسش کوتاه کاربر را به یک عبارت جستجوی دقیق و کوتاه (حداکثر ۶ کلمه شامل نام یا دسته کالا) "
                "برای جستجوی انبار تجهیزات پزشکی تبدیل کنید. فقط عبارت جستجو را بدون هیچ توضیح اضافه‌ای بنویسید."
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        chain = rephrase_prompt | llm_fast | StrOutputParser()
        standalone_q = chain.invoke({"chat_history": chat_history, "input": user_message}).strip()
        if standalone_q:
            return standalone_q
    except Exception:
        pass
    return user_message


def render_product_badges(matches: List[Dict[str, Any]], frontend_url: str = ""):
    """Render product recommendation badges below assistant messages."""
    if not matches:
        return
    st.markdown("##### 🛍️ کالاهای مرتبط در انبار:")
    cols = st.columns(min(len(matches), 4))
    clean_frontend = (frontend_url or "").strip().rstrip("/")
    for idx, item in enumerate(matches[:4]):
        with cols[idx]:
            price_val = item.get("price", 0)
            price_str = f"{to_persian_digits(f'{price_val:,}')} تومان" if price_val else "تماس بگیرید"
            stock_badge = "✅ موجود در انبار" if item.get("in_stock") else "❌ ناموجود"
            raw_url = str(item.get("url") or "")
            if raw_url.startswith("/") and clean_frontend:
                href = f"{clean_frontend}{raw_url}"
            elif raw_url.startswith("http"):
                href = raw_url
            else:
                href = ""

            card_inner = f"""
                <div class="product-name">{item.get('name')}</div>
                <div class="product-price">{price_str} | {stock_badge}</div>
            """
            if href:
                st.markdown(
                    f'<a class="product-badge-card" href="{href}" target="_blank" rel="noopener noreferrer">{card_inner}</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="product-badge-card">{card_inner}</div>',
                    unsafe_allow_html=True,
                )


# ── Load Secrets & Environment Defaults ──────────────────────────────────────
secret_api_key, key_in_secrets = get_secret("OPENAI_API_KEY", "")
secret_base_url, base_in_secrets = get_secret("OPENAI_BASE_URL", "https://api.openai.com/v1")
secret_model, _ = get_secret("OPENAI_MODEL", "gpt-4o-mini")
secret_embed_model, _ = get_secret("OPENAI_EMBED_MODEL", "text-embedding-3-small")
secret_collection_name, _ = get_secret("OPENAI_CHROMA_COLLECTION", "medical_products_openai")
secret_api_base, _ = get_secret("VITE_API_BASE", "https://api.parsinteb.ir")
default_products_api_url = f"{secret_api_base.rstrip('/')}/shop/products/"
secret_products_api_url, products_url_in_secrets = get_secret(
    "PRODUCTS_API_URL",
    default_products_api_url,
)
secret_frontend_url, _ = get_secret("FRONTEND_URL", "https://parsinteb.ir")

# ── Sidebar UI & Diagnostics ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ تنظیمات دستیار هوش مصنوعی (OpenAI)")

    if key_in_secrets:
        openai_api_key = secret_api_key
    else:
        openai_api_key = st.text_input(
            "کلید دسترسی (OPENAI_API_KEY)",
            value=secret_api_key,
            type="password",
            help="کلید API سرویس OpenAI یا سرویس‌های سازگار (قابل تنظیم در Streamlit Secrets)"
        )

    if base_in_secrets or key_in_secrets:
        openai_base_url = secret_base_url
    else:
        openai_base_url = st.text_input(
            "آدرس سرور (OPENAI_BASE_URL)",
            value=secret_base_url,
            help="آدرس پایه API (پیش‌فرض: https://api.openai.com/v1)"
        )

    if products_url_in_secrets:
        products_api_url = secret_products_api_url
    else:
        products_api_url = st.text_input(
            "آدرس API محصولات (PRODUCTS_API_URL)",
            value=secret_products_api_url,
            placeholder="https://api.parsinteb.ir/shop/products/",
            help="آدرس اندپوینت محصولات بک‌اند برای همگام‌سازی خودکار ChromaDB"
        )

    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]
    if secret_model and secret_model not in model_options:
        model_options.insert(0, secret_model)
    default_model_idx = model_options.index(secret_model) if secret_model in model_options else 0

    model_name = st.selectbox(
        "مدل زبانی (OpenAI Model)",
        options=model_options,
        index=default_model_idx
    )

    embed_options = ["text-embedding-3-small", "text-embedding-3-large"]
    if secret_embed_model and secret_embed_model not in embed_options:
        embed_options.insert(0, secret_embed_model)
    default_embed_idx = embed_options.index(secret_embed_model) if secret_embed_model in embed_options else 0

    embed_model = st.selectbox(
        "مدل بردارساز (Embedding Model)",
        options=embed_options,
        index=default_embed_idx
    )

    temperature = st.slider("درجه خلاقیت (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    n_results = st.slider("تعداد کالاهای بررسی تخصصی در هر پاسخ", min_value=2, max_value=8, value=5)

    st.markdown("---")
    st.markdown("### 📊 وضعیت سرویس‌ها")

    if openai_api_key.strip():
        badge_source = " (از Secrets)" if key_in_secrets else ""
        st.success(f"🟢 کلید OpenAI: تنظیم شده است{badge_source}")
    else:
        st.error("🔴 کلید OpenAI: وارد نشده است (`OPENAI_API_KEY`)")

    # Load full active product catalog (live API or bundled JSON fallback)
    all_catalog_products, used_fallback_catalog = get_active_catalog(products_api_url)

    collection = get_chroma_collection_openai(
        str(DEFAULT_CHROMA_DIR),
        openai_api_key,
        openai_base_url,
        embed_model,
        secret_collection_name,
    )

    # Auto-sync ChromaDB if collection is empty or has fewer items than the catalog
    if (
        collection is not None
        and openai_api_key.strip()
        and all_catalog_products
        and not st.session_state.get("initial_api_sync_attempted", False)
    ):
        try:
            if collection.count() < len(all_catalog_products):
                st.session_state["initial_api_sync_attempted"] = True
                with st.spinner("🔄 در حال بردارسازی کامل کاتالوگ محصولات با OpenAI..."):
                    synced_count, sync_err = sync_products_to_chroma(
                        collection,
                        all_catalog_products,
                        secret_frontend_url,
                    )
                    st.session_state["last_sync_error"] = sync_err
        except Exception as e:
            st.session_state["last_sync_error"] = str(e)

    product_count = 0
    if collection is not None:
        try:
            product_count = collection.count()
            if product_count > 0:
                st.success(f"🟢 پایگاه برداری OpenAI: {to_persian_digits(product_count)} محصول ثبت‌شده")
            elif all_catalog_products:
                st.info(f"🟡 کاتالوگ بارگذاری شد ({to_persian_digits(len(all_catalog_products))} محصول)؛ آماده همگام‌سازی برداری.")
            else:
                st.warning("🟡 هیچ محصولی یافت نشد.")
        except Exception:
            st.warning("⚠️ خطا در خواندن پایگاه برداری OpenAI")
    else:
        st.error("🔴 خطا در راه‌اندازی پایگاه برداری ChromaDB.")

    if used_fallback_catalog:
        st.caption("ℹ️ منبع داده: کاتالوگ داخلی مخزن (`products_catalog.json` — ۲۵ محصول پارسین طب)")

    if st.session_state.get("last_sync_error"):
        st.caption(f"⚠️ پیام همگام‌سازی برداری: {st.session_state['last_sync_error']}")

    if products_api_url.strip() or FALLBACK_CATALOG_PATH.exists():
        if st.button("🔄 بروزرسانی و همگام‌سازی محصولات", use_container_width=True):
            get_active_catalog.clear()
            refreshed_products, _ = get_active_catalog(products_api_url)
            with st.spinner("در حال بردارسازی مجدد تمامی محصولات در ChromaDB..."):
                synced_count, sync_err = sync_products_to_chroma(
                    collection,
                    refreshed_products,
                    secret_frontend_url,
                )
                if sync_err and synced_count == 0:
                    st.session_state["last_sync_error"] = sync_err
                    st.error(f"❌ خطا در همگام‌سازی: {sync_err}")
                else:
                    st.session_state["last_sync_error"] = ""
                    st.toast(f"✅ {to_persian_digits(synced_count)} محصول با موفقیت همگام‌سازی شد!")
                    st.rerun()

    st.markdown("---")
    if st.button("🗑️ پاک کردن حافظه گفتگو", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("🔍 تست جستجوی ترکیبی کاتالوگ"):
        search_test = st.text_input("جستجوی مفهومی یا نام محصول:", placeholder="مثال: فشارسنج یا بخور سرد")
        if search_test:
            results = hybrid_search_products(
                collection,
                all_catalog_products,
                search_test,
                n_results=4,
                frontend_url=secret_frontend_url,
            )
            for r in results:
                m = r["metadata"]
                p_val = m.get("price", 0)
                st.caption(f"🛍️ **{m.get('name')}** - {to_persian_digits(f'{p_val:,}')} تومان")


# ── Main Chat Area ───────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🩺 مشاور هوشمند تجهیزات پزشکی پارسین طب (نسخه OpenAI)</h1>
    <p>پاسخ‌گویی تخصصی به سوالات پزشکی، راهنمایی خرید و استعلام دقیق موجودی و قیمت تمامی تجهیزات</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام! من مشاور هوشمند تجهیزات پزشکی پارسین طب هستم. به تمامی محصولات، قیمت‌های روز و مشخصات فنی انبار دسترسی کامل دارم؛ چطور می‌توانم راهنمایی‌تان کنم؟",
            "matches": []
        }
    ]

for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🩺" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("matches"):
            render_product_badges(msg["matches"], secret_frontend_url)

if user_prompt := st.chat_input("پرسش پزشکی یا نام محصول مورد نظر خود را بنویسید..."):
    st.chat_message("user", avatar="👤").markdown(user_prompt)

    history_langchain: List[BaseMessage] = []
    for m in st.session_state.messages[-6:]:
        if m["role"] == "user":
            history_langchain.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history_langchain.append(AIMessage(content=m["content"]))

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    llm_fast = get_llm(openai_api_key, openai_base_url, model_name, temperature=0.0, max_tokens=50)
    search_query = contextualize_query(user_prompt, history_langchain, llm_fast)

    matched_products = hybrid_search_products(
        collection,
        all_catalog_products,
        search_query,
        n_results=n_results,
        frontend_url=secret_frontend_url,
    )
    matches_metadata = [m.get("metadata", {}) for m in matched_products if m.get("metadata")]

    context_docs = [m.get("document", "") for m in matched_products if m.get("document")]
    detailed_matches_context = "\n\n---\n\n".join(context_docs) if context_docs else "مورد تخصصی مستقیمی یافت نشد."
    full_inventory_overview = build_catalog_overview_index(all_catalog_products)

    system_instruction = (
        "شما «مشاور هوشمند و تخصصی تجهیزات پزشکی پارسین طب (مدی کالا)» هستید.\n"
        "وظیفه شما راهنمایی دقیق، علمی و کاملاً مستند به کاتالوگ واقعی فروشگاه است.\n\n"
        "قوانین بسیار مهم (حتماً رعایت کنید):\n"
        "۱. **صرفاً بر اساس اطلاعات واقعی زیر پاسخ دهید**؛ هرگز نام محصول، برند، ویژگی فنی یا قیمتی را از خودتان حدس نزنید یا نسازید.\n"
        "۲. شما به «فهرست کامل تمامی کالاهای موجود در فروشگاه» و همچنین «مشخصات فنی کامل کالاهای مرتبط» دسترسی دارید. اگر کاربر درباره دسته‌ای از محصولات (مثل همه فشارسنج‌ها، همه بخورها، شیردوش‌ها، ماساژورها یا ارزان‌ترین/گران‌ترین کالا) پرسید، **همه موارد مرتبط موجود در فهرست کامل فروشگاه** را به طور دقیق معرفی کنید.\n"
        "۳. نام دقیق کالا، برند، وضعیت موجودی و قیمت دقیق به تومان را عیناً مطابق اطلاعات درج‌شده بیان کنید.\n"
        "۴. نیازی به نوشتن لینک نیست؛ کارت‌های محصولات به صورت خودکار زیر پاسخ نمایش داده می‌شوند.\n"
        "۵. ارسال سفارش‌ها در شهر رشت فوری و برای سایر شهرها با پست پیشتاز انجام می‌شود.\n"
        "۶. اگر کالای درخواستی کاربر اصلاً در فهرست کل محصولات فروشگاه وجود نداشت، صریحاً اعلام کنید که فعلاً در انبار موجود نیست و نزدیک‌ترین کالاهای موجود در فهرست را پیشنهاد دهید.\n\n"
        f"=== بخش اول: مشخصات فنی کامل کالاهای منتخب مرتبط با پرسش ===\n{detailed_matches_context}\n\n"
        f"=== بخش دوم: فهرست کامل تمامی محصولات، قیمت‌ها و موجودی فروشگاه پارسین طب ===\n{full_inventory_overview}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    llm = get_llm(openai_api_key, openai_base_url, model_name, temperature=temperature, max_tokens=650)
    chain = qa_prompt | llm

    with st.chat_message("assistant", avatar="🩺"):
        def generate_response():
            try:
                for chunk in chain.stream({"chat_history": history_langchain, "input": user_prompt}):
                    yield chunk.content
            except Exception as err:
                logger.error(f"[Streamlit OpenAI RAG] Stream error: {err}")
                yield f"⚠️ خطا در برقراری ارتباط با سرویس OpenAI: {err}\n\nلطفاً از صحت `OPENAI_API_KEY` اطمینان حاصل فرمایید."

        full_reply = st.write_stream(generate_response())

        if matches_metadata:
            render_product_badges(matches_metadata, secret_frontend_url)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_reply,
        "matches": matches_metadata
    })
