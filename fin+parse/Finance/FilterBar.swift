// FilterBar.swift — Shared filter controls for all dashboard pages.

import SwiftUI

struct FilterBar: View {
    @ObservedObject var service: DashboardService

    var body: some View {
        VStack(spacing: 10) {
            // Period selector
            HStack(spacing: 4) {
                ForEach(DashboardPeriod.allCases) { p in
                    Button(action: { withAnimation { service.period = p } }) {
                        Text(p.label)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(service.period == p ? .white : AppTheme.textMuted)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .background(service.period == p ? AppTheme.accent : AppTheme.surface2)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .buttonStyle(PressEffect())
                }
            }

            HStack(spacing: 8) {
                // Bank picker
                Menu {
                    ForEach(DashboardBankFilter.allCases) { b in
                        Button(action: { service.bankFilter = b }) {
                            Label(b.label, systemImage: service.bankFilter == b ? "checkmark" : "")
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "building.columns")
                            .font(.system(size: 11))
                        Text(service.bankFilter.label)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(service.bankFilter != .all ? AppTheme.accent : AppTheme.textMuted)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(service.bankFilter != .all ? AppTheme.accentSoft : AppTheme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                // Type picker
                Menu {
                    ForEach(DashboardTypeFilter.allCases) { t in
                        Button(action: { service.typeFilter = t }) {
                            Label(t.label, systemImage: service.typeFilter == t ? "checkmark" : "")
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.left.arrow.right")
                            .font(.system(size: 11))
                        Text(service.typeFilter.label)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(service.typeFilter != .all ? AppTheme.accent : AppTheme.textMuted)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(service.typeFilter != .all ? AppTheme.accentSoft : AppTheme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                Spacer()
            }
        }
        .padding(.horizontal, 20)
    }
}
