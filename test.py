#!/usr/bin/env python3
"""
Script to insert voice agents into Railway PostgreSQL database
Using your exact database credentials
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Your Railway database URL
DATABASE_URL = "postgresql://postgres:CVvYMUrUMEUtknNEOLmDaRTyGBsRNgrd@ballast.proxy.rlwy.net:13609/railway"


def create_voice_agents_table():
    """Create voice_agents table if it doesn't exist"""

    # Create engine
    engine = create_engine(DATABASE_URL)

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS voice_agents (
        id SERIAL PRIMARY KEY,
        topic VARCHAR NOT NULL,
        title VARCHAR NOT NULL,
        description VARCHAR NOT NULL,
        image_url VARCHAR NOT NULL,
        agent_id VARCHAR NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """

    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
            print("✅ voice_agents table created/verified")
        return engine
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return None


def insert_agents(engine):
    """Insert your voice agents with ElevenLabs IDs"""

    agents_data = [
        {
            "topic": "cars",
            "title": "Car Expert",
            "description": "Discuss everything about automobiles, engines, driving techniques, maintenance, and automotive industry news",
            "image_url": "https://images.unsplash.com/photo-1550355291-bbee04a92027?w=400",
            "agent_id": "agent_3201k0xj8t6jfra8na00s910sah7"
        },
        {
            "topic": "travel",
            "title": "Travel Guide",
            "description": "Explore destinations, cultural insights, travel planning, transportation, accommodation, and local cuisine",
            "image_url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400",
            "agent_id": "agent_3901k0xawmnpermr2vgy4d6ewza3"
        },
        {
            "topic": "football",
            "title": "Football Coach",
            "description": "Master soccer tactics, formations, player analysis, match predictions, training techniques, and football history",
            "image_url": "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=400",
            "agent_id": "agent_01k0rx0fe8fz6tfyj1n5w7ek80"
        }
    ]

    try:
        with engine.connect() as conn:
            # Check if agents already exist
            result = conn.execute(text("SELECT COUNT(*) FROM voice_agents"))
            existing_count = result.scalar()

            if existing_count > 0:
                print(f"⚠️  Found {existing_count} existing voice agents")

                # Show existing agents
                existing = conn.execute(text("SELECT topic, title, agent_id FROM voice_agents ORDER BY topic"))
                print("\nExisting agents:")
                for agent in existing.fetchall():
                    print(f"  - {agent.topic}: {agent.title} ({agent.agent_id[:20]}...)")

                choice = input(
                    "\nChoose action:\n1. Skip (keep existing)\n2. Update existing\n3. Delete and recreate\nEnter choice (1/2/3): ").strip()

                if choice == "1":
                    print("✅ Keeping existing agents")
                    return True
                elif choice == "2":
                    # Update existing agents
                    for agent_data in agents_data:
                        update_sql = """
                        UPDATE voice_agents 
                        SET title = :title, description = :description, 
                            image_url = :image_url, agent_id = :agent_id, is_active = true
                        WHERE topic = :topic
                        """
                        result = conn.execute(text(update_sql), agent_data)
                        if result.rowcount > 0:
                            print(f"✅ Updated {agent_data['topic']} agent")
                        else:
                            # Agent doesn't exist, insert it
                            insert_sql = """
                            INSERT INTO voice_agents (topic, title, description, image_url, agent_id, is_active)
                            VALUES (:topic, :title, :description, :image_url, :agent_id, true)
                            """
                            conn.execute(text(insert_sql), agent_data)
                            print(f"✅ Created new {agent_data['topic']} agent")

                    conn.commit()
                    print("🎉 Agents updated successfully!")
                    return True

                elif choice == "3":
                    # Delete and recreate
                    conn.execute(text("DELETE FROM voice_agents"))
                    print("🗑️  Deleted existing agents")
                else:
                    print("❌ Invalid choice")
                    return False

            # Insert new agents
            insert_sql = """
            INSERT INTO voice_agents (topic, title, description, image_url, agent_id, is_active)
            VALUES (:topic, :title, :description, :image_url, :agent_id, true)
            """

            for agent_data in agents_data:
                conn.execute(text(insert_sql), agent_data)
                print(f"✅ Created {agent_data['topic']} agent: {agent_data['title']}")

            conn.commit()
            print("\n🎉 All voice agents created successfully!")
            return True

    except Exception as e:
        print(f"❌ Error inserting agents: {e}")
        return False


def verify_agents(engine):
    """Verify the agents were created correctly"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, topic, title, agent_id, is_active, created_at 
                FROM voice_agents 
                ORDER BY topic
            """))

            agents = result.fetchall()

            if not agents:
                print("❌ No agents found in database")
                return False

            print(f"\n📊 Database contains {len(agents)} voice agents:")
            print("-" * 80)

            for agent in agents:
                status = "🟢 Active" if agent.is_active else "🔴 Inactive"
                print(f"{status} [{agent.id}] {agent.topic.upper()}: {agent.title}")
                print(f"    Agent ID: {agent.agent_id}")
                print(f"    Created: {agent.created_at}")
                print()

            return True

    except Exception as e:
        print(f"❌ Error verifying agents: {e}")
        return False


def main():
    print("🎙️ Voice Agents Database Setup")
    print("Railway PostgreSQL Database")
    print("=" * 50)

    # Test connection and create table
    print("🔌 Connecting to Railway database...")
    engine = create_voice_agents_table()

    if not engine:
        print("❌ Failed to connect to database")
        return

    print("✅ Database connection successful!")

    # Insert agents
    print("\n📝 Setting up voice agents...")
    if insert_agents(engine):
        print("\n🔍 Verifying results...")
        verify_agents(engine)

    print("\n✨ Setup complete!")
    print("\nYour ElevenLabs agents are now saved in the database:")
    print("🚗 Cars: agent_3201k0xj8t6jfra8na00s910sah7")
    print("✈️ Travel: agent_3901k0xawmnpermr2vgy4d6ewza3")
    print("⚽ Football: agent_01k0rx0fe8fz6tfyj1n5w7ek80")


if __name__ == "__main__":
    main()