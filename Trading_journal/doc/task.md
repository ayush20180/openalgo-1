# Tasks for Dark/Light Mode Theme Switching

## Phase 1: Core Theme Switching Logic

- [x] **Task 1.1: Add Theme Storage (`dcc.Store`)**
  - [x] Sub-task 1.1.1: Add a `dcc.Store` component to the main layout in `app/main.py` with `id='theme-store'` and `storage_type='local'` (to persist across sessions). Initialize with `'light'`.
- [x] **Task 1.2: Add Theme Stylesheet Link (`html.Link`)**
  - [x] Sub-task 1.2.1: Add an `html.Link` component to the main layout in `app/main.py` with `id='theme-stylesheet-link'`, `rel='stylesheet'`, and an initial `href` (e.g., `dbc.themes.BOOTSTRAP`).
  - [x] Sub-task 1.2.2: Ensure the Dash app initialization in `app/main.py` does *not* pre-load a theme via `external_stylesheets` if this `html.Link` method is used to manage themes dynamically. The `html.Link` will be the primary source of the switchable theme.
- [x] **Task 1.3: Add Theme Toggle UI (`dbc.Switch`)**
  - [x] Sub-task 1.3.1: Add a `dbc.Switch` component to the main layout in `app/main.py` (e.g., in the header/navbar) with `id='theme-switch'`, `value=False` (for light theme initially).
- [x] **Task 1.4: Implement Callbacks for Theme Switching**
  - [x] Sub-task 1.4.1: Create a callback in `app/main.py` that:
      - Takes `Input('theme-switch', 'value')`.
      - Outputs `Output('theme-store', 'data')`.
      - Updates the store to 'dark' if switch is `True`, 'light' if `False`.
  - [x] Sub-task 1.4.2: Create a callback in `app/main.py` that:
      - Takes `Input('theme-store', 'data')`.
      - Outputs `Output('theme-stylesheet-link', 'href')`.
      - Sets `href` to `dbc.themes.DARKLY` if store is 'dark', and `dbc.themes.BOOTSTRAP` if store is 'light'.
  - [ ] Sub-task 1.4.3 (Optional): Create a callback to update the label/icon of the `dbc.Switch` based on the theme.
  - [x] Sub-task 1.4.4: Enhance theme switching for CSS targeting:
    - [x] Add `id='app-container'` to the main `dbc.Container` in `app/main.py`.
    - [x] Modify `update_theme_stylesheet` callback in `app/main.py` to also output `className` to `app-container` (e.g., 'app-container-light'/'app-container-dark').

## Phase 2: UI Integration and Styling

- [x] **Task 2.1: Position Theme Toggle**
  - [x] Sub-task 2.1.1: Determine the best location for the theme toggle in the UI (e.g., top navigation bar) and update `app/main.py` layout accordingly.
- [ ] **Task 2.2: Test Theme Application (Blocked - `run_command` failed)**
  - [ ] Sub-task 2.2.1: Thoroughly test the theme switching across all tabs and components to ensure consistent application.
  - [ ] Sub-task 2.2.2: Verify that Plotly graphs adapt correctly or if they need specific template adjustments for dark/light themes. (Plotly has built-in theme templates like 'plotly_dark'). This might require an additional callback to update graph layouts.

## Phase 3: Documentation and Refinement

- [ ] **Task 3.1: Update `task.md` (Meta-task)**
  - [ ] Sub-task 3.1.1: Mark relevant tasks in this file (`doc/task.md`) as complete as they are finished.
- [ ] **Task 3.2: Update `structure.md` (if necessary)**
  - [ ] Sub-task 3.2.1: If any significant structural changes or new conventions are introduced, update `doc/structure.md`.
- [x] **Task 3.3: Update `USER_GUIDE.md`**
  - [x] Sub-task 3.3.1: Add instructions for users on how to use the new theme toggle in `USER_GUIDE.md`.

## Phase 4: UI/UX Overhaul (TradeZella-like Dashboard)

### Sub-Phase 4.1: Structural Refactoring & Basic Layout
- [x] **Task 4.1.1: Modify `app/main.py` for Two-Column Layout**
  - [x] Sub-task 4.1.1.1: Implement the basic structure with a left sidebar column and a main content column.
- [x] **Task 4.1.2: Create Sidebar Navigation**
  - [x] Sub-task 4.1.2.1: Develop the sidebar component (e.g., in `app/utils/sidebar.py` or directly in `main.py`) using `dbc.Nav` with links for each main section.
  - [x] Sub-task 4.1.2.2: Implement callbacks to display the correct content in the main area based on sidebar navigation clicks. Initially, this will involve showing/hiding existing content blocks.
- [x] **Task 4.1.3: Reposition File Upload**
  - [x] Sub-task 4.1.3.1: Choose and implement the new location for the `dcc.Upload` component.
- [x] **Task 4.1.4: Implement URL Routing Callback**
  - [x] Sub-task 4.1.4.1: Create a callback in `app/main.py` to display content based on URL pathname.

### Sub-Phase 4.2: Content Refactoring into Cards & Initial Styling
- [x] **Task 4.2.1: Refactor Section Content into `dbc.Card`s**
  - [x] Sub-task 4.2.1.1: Systematically go through each section (Overview, Algorithm Analysis, etc.) and wrap its constituent elements (metrics, graphs, tables) in `dbc.Card` components.
- [x] **Task 4.2.2: Basic Custom CSS Styling**
  - [x] Sub-task 4.2.2.1: Create `assets/custom.css`.
  - [x] Sub-task 4.2.2.2: Apply initial custom styles for the sidebar, cards, and overall layout (margins, padding, basic colors).

### Sub-Phase 4.3: Advanced Styling, Polish & Iconography
- [x] **Task 4.3.1: Refine Typography and Color Palette**
  - [x] Sub-task 4.3.1.1: Select and apply appropriate fonts globally via `custom.css`.
  - [x] Sub-task 4.3.1.2: Finalize and apply a consistent color scheme that works well with both light and dark themes.
- [x] **Task 4.3.2: Enhance Component Styling**
  - [x] Sub-task 4.3.2.1: Apply detailed styling to tables, graphs, form elements for a cohesive look.
- [x] **Task 4.3.3: Integrate Icons**
  - [x] Sub-task 4.3.3.1: Add icons to sidebar navigation links and other relevant UI elements.
- [ ] **Task 4.3.4: Responsiveness Testing (Blocked - `run_command` failed)**
  - [ ] Sub-task 4.3.4.1: Ensure the new layout is responsive and functions well on different screen sizes (Blocked - `run_command` failed).

### Sub-Phase 4.4: Global Filter Integration & Final Review
- [x] **Task 4.4.1: Implement Global Filter Placement**
  - [x] Sub-task 4.4.1.1: Decide on the final location for global filters (e.g., collapsible sidebar section) and implement it.
  - [x] Sub-task 4.4.1.2: Style the global filters to match the new UI.
- [ ] **Task 4.4.2: Comprehensive Testing**
  - [ ] Sub-task 4.4.2.1: Thoroughly test all application functionalities with the new UI.
  - [ ] Sub-task 4.4.2.2: Perform cross-browser compatibility checks.
- [ ] **Task 4.4.3: Documentation Update for UI Overhaul**
  - [ ] Sub-task 4.4.3.1: Update `doc/structure.md` to reflect the new UI structure and any new components/modules.
  - [ ] Sub-task 4.4.3.2: Update `USER_GUIDE.md` with new screenshots and instructions for the redesigned interface.