**Feature: UI/UX Overhaul for TradeZella-like Dashboard**

**1. Introduction**
   This document outlines the requirements and plan for a significant UI/UX overhaul of the Algorithmic Trading Dashboard. The goal is to modernize the interface, making it more professional, data-centric, and visually appealing, drawing inspiration from platforms like TradeZella.

**2. Goals**
   - Achieve a modern, professional, and intuitive user interface.
   - Improve information hierarchy and data presentation.
   - Enhance visual appeal through better use of space, typography, and color schemes.
   - Reorganize navigation for better usability (e.g., sidebar navigation).
   - Ensure the redesigned UI is responsive and user-friendly.

**3. Current UI Structure (Summary)**
   - Main layout in `app/main.py` using `dbc.Container`.
   - Simple header with title and theme switch.
   - `dcc.Upload` for data input.
   - A distinct `global_filters.py` section.
   - `dbc.Tabs` for navigating between: Overview, Algorithm Analysis, Advanced Analytics, Trade Details, and Journal Management.
   - Content for each tab is defined within `app/main.py` or imported from utility modules like `advanced_dashboard.py` and `journal_management.py`.

**4. Proposed Redesign Inspired by TradeZella**

   **4.1. Overall Layout Philosophy:**
      - **Modern & Clean:** Focus on clarity, minimalism, and professional aesthetics.
      - **Data-Centric:** Prioritize the display and accessibility of key trading data and analytics.
      - **Component-Based:** Utilize cards and well-defined sections for organizing information.

   **4.2. Key Redesign Areas & Proposed Changes:**
      *   **Navigation:**
          *   **Current:** `dbc.Tabs` for main sections.
          *   **Proposed:** Implement a persistent left sidebar (`dbc.Nav` within a styled `dbc.Col`) for primary navigation. This sidebar will contain links to 'Overview', 'Algorithm Analysis', 'Advanced Analytics', 'Trade Details', 'Journal Management', and potentially 'Upload Data'.
      *   **Main Content Area:**
          *   **Current:** Content displayed directly under tabs.
          *   **Proposed:** The area to the right of the sidebar will display the content corresponding to the selected sidebar navigation item. Content within each section (metrics, graphs, tables) will be refactored into `dbc.Card` components for better visual organization and a dashboard feel.
      *   **Header:**
          *   **Current:** Contains app title and theme switch.
          *   **Proposed:** Streamline the header to primarily feature the application title and theme switch. The file upload might be moved or given a more prominent, integrated spot.
      *   **File Upload (`dcc.Upload`):**
          *   **Current:** A distinct section below the header.
          *   **Proposed:** Integrate more seamlessly. Options:
              1.  A dedicated 'Upload Data' item in the new sidebar.
              2.  A prominent button/area in the header or at the top of the 'Overview' page (default landing page).
              3.  The initial dashboard state (pre-upload) should clearly guide the user to upload data.
      *   **Global Filters (`global_filters.py`):**
          *   **Current:** A separate, always visible section.
          *   **Proposed:** Integrate more cohesively. Options:
              1.  A collapsible section within the sidebar (always accessible but can be hidden).
              2.  A dedicated filter bar at the top of the main content area for each view (consistent styling needed).
              A collapsible sidebar section is preferred to keep the main content area cleaner.
      *   **Visual Appeal & Styling:**
          *   **Themes:** Augment existing light/dark DBC themes or explore alternatives (e.g., `LUX`, `QUARTZ` for dark; `FLATLY`, `SANDSTONE` for light) for a more modern base.
          *   **Custom CSS:** Create `assets/custom.css` for fine-grained control over spacing, typography (clean, readable fonts), card styling, sidebar appearance, and overall look-and-feel.
          *   **Component Styling:** Ensure consistent and enhanced styling for `dbc.Card`, `dash_table.DataTable`, `dcc.Graph`, and form elements.
          *   **Iconography:** Introduce icons (e.g., Font Awesome via DBC) for sidebar navigation links and action buttons to improve visual communication.

**5. Implementation Phases and Tasks**

   **Phase 1: Structural Refactoring & Basic Layout**
     *   **Task 1.1: Modify `app/main.py` for Two-Column Layout**
         *   Sub-task 1.1.1: Implement the basic structure with a left sidebar column and a main content column.
     *   **Task 1.2: Create Sidebar Navigation**
         *   Sub-task 1.2.1: Develop the sidebar component (e.g., in `app/utils/sidebar.py` or directly in `main.py`) using `dbc.Nav` with links for each main section.
         *   Sub-task 1.2.2: Implement callbacks to display the correct content in the main area based on sidebar navigation clicks. Initially, this will involve showing/hiding existing content blocks.
     *   **Task 1.3: Reposition File Upload**
         *   Sub-task 1.3.1: Choose and implement the new location for the `dcc.Upload` component.

   **Phase 2: Content Refactoring into Cards & Initial Styling**
     *   **Task 2.1: Refactor Section Content into `dbc.Card`s**
         *   Sub-task 2.1.1: Systematically go through each section (Overview, Algorithm Analysis, etc.) and wrap its constituent elements (metrics, graphs, tables) in `dbc.Card` components.
     *   **Task 2.2: Basic Custom CSS Styling**
         *   Sub-task 2.2.1: Create `assets/custom.css`.
         *   Sub-task 2.2.2: Apply initial custom styles for the sidebar, cards, and overall layout (margins, padding, basic colors).

   **Phase 3: Advanced Styling, Polish & Iconography**
     *   **Task 3.1: Refine Typography and Color Palette**
         *   Sub-task 3.1.1: Select and apply appropriate fonts globally via `custom.css`.
         *   Sub-task 3.1.2: Finalize and apply a consistent color scheme that works well with both light and dark themes.
     *   **Task 3.2: Enhance Component Styling**
         *   Sub-task 3.2.1: Apply detailed styling to tables, graphs, form elements for a cohesive look.
     *   **Task 3.3: Integrate Icons**
         *   Sub-task 3.3.1: Add icons to sidebar navigation links and other relevant UI elements.
     *   **Task 3.4: Responsiveness Testing**
         *   Sub-task 3.4.1: Ensure the new layout is responsive and functions well on different screen sizes.

   **Phase 4: Global Filter Integration & Final Review**
     *   **Task 4.1: Implement Global Filter Placement**
         *   Sub-task 4.1.1: Decide on the final location for global filters (e.g., collapsible sidebar section) and implement it.
         *   Sub-task 4.1.2: Style the global filters to match the new UI.
     *   **Task 4.2: Comprehensive Testing**
         *   Sub-task 4.2.1: Thoroughly test all application functionalities with the new UI.
         *   Sub-task 4.2.2: Perform cross-browser compatibility checks.
     *   **Task 4.3: Documentation Update**
         *   Sub-task 4.3.1: Update `doc/structure.md` to reflect the new UI structure and any new components/modules.
         *   Sub-task 4.3.2: Update `USER_GUIDE.md` with new screenshots and instructions for the redesigned interface.

**6. Considerations**
   - **Callback Complexity:** Refactoring the layout might require adjustments to existing callbacks, particularly those controlling visibility or updating components that are being moved or restructured.
   - **Modularity:** Aim to keep new UI components (like the sidebar) modular, potentially in their own utility files if they become complex.
   - **Performance:** Ensure that the new design and any added CSS/JS do not negatively impact application loading time or performance.

**7. Future Enhancements (Out of Scope for this PRD)**
   - More advanced dashboard customization options for the user.
   - User-selectable accent colors or more granular theme controls.