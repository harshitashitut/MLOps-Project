import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://qftexohgrwvlsisiccmx.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmdGV4b2hncnd2bHNpc2ljY214Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMzY5NjAsImV4cCI6MjA4MDYxMjk2MH0.6WzG5x3-oev6ySmwk0skaTxjHdSK1zUiaq2hkm8DyN0'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)