# gallery.py
"""
Image Gallery Endpoint for SD.CPP
Renders an HTML gallery of all generated images/videos with pagination and limit selector.
"""
import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

OUTPUT_DIR = "/sdcpp/outputs"  # Ensure this matches your actual output directory

# Allowed limits and default
ALLOWED_LIMITS = [5, 12, 50, 100, 200, "all"]
DEFAULT_LIMIT = 12

@router.get("/gallery", response_class=HTMLResponse)
async def image_gallery(request: Request):
    """Display gallery of generated images with pagination and limit selection"""
    # Get query parameters safely
    page_param = request.query_params.get("page", "1")
    limit_param = request.query_params.get("limit", str(DEFAULT_LIMIT))

    # Parse page number
    try:
        page_num = max(1, int(page_param))
    except ValueError:
        page_num = 1

    # Parse limit
    if limit_param == "all":
        limit = "all"
    else:
        try:
            limit = int(limit_param)
            if limit not in ALLOWED_LIMITS:
                limit = DEFAULT_LIMIT
        except ValueError:
            limit = DEFAULT_LIMIT

    # Get all valid files and sort by creation time (newest first)
    files_with_time = []
    for f in os.listdir(OUTPUT_DIR):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4")):
            file_path = os.path.join(OUTPUT_DIR, f)
            try:
                ctime = os.path.getctime(file_path)
                files_with_time.append((f, ctime))
            except Exception:
                files_with_time.append((f, 0))

    # Sort by creation time (descending)
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    all_files = [f[0] for f in files_with_time]
    total_items = len(all_files)

    # Apply limit and pagination
    if limit == "all":
        displayed_files = all_files
        total_pages = 1
        page_num = 1  # Always show page 1 when "all"
    else:
        # Calculate total pages based on ALL files
        total_pages = max(1, (total_items + limit - 1) // limit)
        # Clamp page number to valid range
        page_num = min(page_num, total_pages)
        # Calculate slice for current page
        start_idx = (page_num - 1) * limit
        end_idx = start_idx + limit
        displayed_files = all_files[start_idx:end_idx]

    # Pagination navigation
    prev_page = page_num - 1 if page_num > 1 else None
    next_page = page_num + 1 if page_num < total_pages else None

    # Build limit selector dropdown
    limit_options = ""
    for opt in ALLOWED_LIMITS:
        selected = "selected" if str(opt) == limit_param else ""
        label = "All" if opt == "all" else str(opt)
        limit_options += f'<option value="{opt}" {selected}>{label}</option>'

    # Build pagination controls
    pagination_html = ""
    # Always show the dropdown
    pagination_html = f"""
    <div style="margin:24px 0; text-align:center; font-size:1rem; display:flex; flex-wrap:wrap; gap:12px; justify-content:center; align-items:center;">
      <label style="color:#ccc; margin-right:8px;">Items per page:</label>
      <select onchange="location.href=`?page=1&limit=${{this.value}}`" style="padding:6px 12px; border:1px solid #444; background:#222; color:#ccc; border-radius:4px; font-size:1rem;">
        {limit_options}
      </select>
    """

    if limit != "all":
        # Only show pagination if there are multiple pages
        if total_pages > 1:
            # Always include limit in URLs
            prev_url = f"?page={prev_page}&limit={limit}" if prev_page else None
            next_url = f"?page={next_page}&limit={limit}" if next_page else None

            # Only show buttons if they lead somewhere
            prev_button = f'<a href="{prev_url}" style="margin:0 4px; padding:6px 12px; border:1px solid #444; background:#222; color:#ccc; text-decoration:none; border-radius:4px;">← Prev</a>' if prev_page else ''
            next_button = f'<a href="{next_url}" style="margin:0 4px; padding:6px 12px; border:1px solid #444; background:#222; color:#ccc; text-decoration:none; border-radius:4px;">Next →</a>' if next_page else ''

            pagination_html += f"""
              {prev_button}
              <span style="margin:0 8px; color:#aaa;">Page {page_num} of {total_pages}</span>
              {next_button}
            </div>
            """
        else:
            # Only one page, no pagination needed
            pagination_html += f"""
              <span style="margin:0 8px; color:#aaa;">Page {page_num} of {total_pages}</span>
            </div>
            """
    else:
        pagination_html += f"""
          <span style="margin:0 8px; color:#aaa;">Showing all {total_items} files</span>
        </div>
        """

    # Generate gallery items
    items = []
    for name in displayed_files:
        url = f"/images/{name}"
        is_video = name.lower().endswith(".mp4")
        thumb_style = "width:100%;height:200px;object-fit:cover;"
        media_tag = (
            f"<video src='{url}' muted loop playsinline style='{thumb_style}'></video>"
            if is_video
            else f"<img src='{url}' loading='lazy' style='{thumb_style}'>"
        )
        items.append(
            f"""
            <a href="{url}" target="_blank" style="text-decoration:none;">
              <div style="margin:8px;border-radius:8px;border:1px solid #444;overflow:hidden;background:#111;">
                {media_tag}
                <div style="padding:6px 8px;font-size:11px;color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                  {name}
                </div>
              </div>
            </a>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>SD.CPP Gallery</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body {{background:#050816;color:#eee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:20px;}}
        .header {{text-align:center;margin-bottom:24px;}}
        .header h1 {{margin:0;font-size:2rem;}}
        .grid {{display:grid;grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));gap:12px;max-width:1400px;margin:0 auto;}}
        .refresh {{position:fixed;right:16px;top:16px;padding:8px 16px;border-radius:999px;border:none;background:#4ecdc4;color:#000;font-weight:600;cursor:pointer;}}
        .refresh:hover {{background:#45b7d1;}}
        select {{appearance:none; background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='6' viewBox='0 0 12 6'%3E%3Cpath fill='%23ccc' d='M0 0h12l-6 6z'/%3E%3C/svg%3E%0A"); background-repeat:no-repeat; background-position:right 10px center; background-size:12px;}}
      </style>
    </head>
    <body>
      <button class="refresh" onclick="location.reload()">Refresh</button>
      <div class="header">
        <h1>SD.CPP Gallery</h1>
        <div style="margin-top:4px;font-size:0.9rem;color:#aaa;">
          Showing {len(displayed_files)} of {total_items} files from /outputs
        </div>
        <div class="top-links" style="margin-top:8px;">
          <a href="/">Gradio UI</a>
          <a href="/docs">API Docs</a>
        </div>
      </div>
      {pagination_html}
      <div class="grid">
        {''.join(items)}
      </div>
      <script>setTimeout(() => location.reload(), 30000);</script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
