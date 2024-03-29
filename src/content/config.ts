import { defineCollection, z } from "astro:content";
import { SITE } from "../consts";

const docs = defineCollection({
  schema: z.object({
    title: z.string().default(SITE.title),
    description: z.string().default(SITE.description),
    date: z.string().optional(),
    platform: z.string().optional(),
    channelId: z.string().optional(),
    embedId: z.string().optional(),
    sourceUrl: z.string().optional(),
    viewes: z.number().optional(),
    likes: z.number().optional(),
    transcript: z.string().optional(),
    lang: z.literal("en-us").default(SITE.defaultLanguage),
    dir: z.union([z.literal("ltr"), z.literal("rtl")]).default("ltr"),
    image: z
      .object({
        src: z.string(),
        alt: z.string().default(""),
      })
      .optional(),
    ogLocale: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = { docs };
