
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_53_8_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0xc08                                   // 000000001708: 9604FF0F 00000C08
	s_mul_i32 s6, s14, 56                                      // 000000001710: 9606B80E
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	s_mov_b32 s12, s13                                         // 000000001718: BE8C000D
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000171C: 84848204
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s8, s2, s4                                       // 000000001728: 80080402
	s_addc_u32 s5, s3, s5                                      // 00000000172C: 82050503
	s_ashr_i32 s7, s6, 31                                      // 000000001730: 86079F06
	s_mul_i32 s4, s13, 3                                       // 000000001734: 9604830D
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001738: 84828206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000173C: BF8704B9
	s_add_u32 s6, s8, s2                                       // 000000001740: 80060208
	s_addc_u32 s7, s5, s3                                      // 000000001744: 82070305
	s_ashr_i32 s5, s4, 31                                      // 000000001748: 86059F04
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000174C: 84828204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001750: BF870009
	s_add_u32 s2, s6, s2                                       // 000000001754: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001758: 82030307
	s_clause 0x4                                               // 00000000175C: BF850004
	s_load_b128 s[4:7], s[2:3], null                           // 000000001760: F4080101 F8000000
	s_load_b32 s13, s[2:3], 0x10                               // 000000001768: F4000341 F8000010
	s_load_b128 s[8:11], s[2:3], 0x70                          // 000000001770: F4080201 F8000070
	s_load_b32 s20, s[2:3], 0x80                               // 000000001778: F4000501 F8000080
	s_load_b128 s[16:19], s[2:3], 0xe0                         // 000000001780: F4080401 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 000000001788: BF89FC07
	v_max_f32_e64 v0, s4, s4                                   // 00000000178C: D5100000 00000804
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001794: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 000000001798: 200000FF FF800000
	v_max3_f32 v0, s6, s5, v0                                  // 0000000017A0: D61C0000 04000A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017A8: BF8700C1
	v_max3_f32 v0, s13, s7, v0                                 // 0000000017AC: D61C0000 04000E0D
	s_clause 0x1                                               // 0000000017B4: BF850001
	s_load_b32 s13, s[2:3], 0xf0                               // 0000000017B8: F4000341 F80000F0
	s_load_b128 s[4:7], s[2:3], 0x150                          // 0000000017C0: F4080101 F8000150
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017C8: D61C0000 04001009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D0: BF870091
	v_max3_f32 v0, s11, s10, v0                                // 0000000017D4: D61C0000 0400140B
	v_max3_f32 v0, s16, s20, v0                                // 0000000017DC: D61C0000 04002810
	s_clause 0x1                                               // 0000000017E4: BF850001
	s_load_b32 s20, s[2:3], 0x160                              // 0000000017E8: F4000501 F8000160
	s_load_b128 s[8:11], s[2:3], 0x1c0                         // 0000000017F0: F4080201 F80001C0
	s_mul_i32 s16, s15, 0x1a8                                  // 0000000017F8: 9610FF0F 000001A8
	s_load_b32 s15, s[2:3], 0x1d0                              // 000000001800: F40003C1 F80001D0
	v_max3_f32 v0, s18, s17, v0                                // 000000001808: D61C0000 04002212
	s_ashr_i32 s17, s16, 31                                    // 000000001810: 86119F10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001814: BF8700A9
	s_lshl_b64 s[2:3], s[16:17], 2                             // 000000001818: 84828210
	s_waitcnt lgkmcnt(0)                                       // 00000000181C: BF89FC07
	v_max3_f32 v0, s13, s19, v0                                // 000000001820: D61C0000 0400260D
	s_add_u32 s2, s0, s2                                       // 000000001828: 80020200
	s_addc_u32 s3, s1, s3                                      // 00000000182C: 82030301
	s_lshl_b32 s0, s14, 3                                      // 000000001830: 8400830E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001834: BF8704A1
	v_max3_f32 v0, s5, s4, v0                                  // 000000001838: D61C0000 04000805
	s_ashr_i32 s1, s0, 31                                      // 000000001840: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001844: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001848: BF8700C1
	v_max3_f32 v0, s7, s6, v0                                  // 00000000184C: D61C0000 04000C07
	s_add_u32 s2, s2, s0                                       // 000000001854: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001858: 82030103
	s_ashr_i32 s13, s12, 31                                    // 00000000185C: 860D9F0C
	v_max3_f32 v0, s8, s20, v0                                 // 000000001860: D61C0000 04002808
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001868: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000186C: BF8700A9
	s_add_u32 s0, s2, s0                                       // 000000001870: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001874: 82010103
	v_max3_f32 v0, s10, s9, v0                                 // 000000001878: D61C0000 0400120A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001880: BF870001
	v_max3_f32 v0, s15, s11, v0                                // 000000001884: D61C0000 0400160F
	global_store_b32 v1, v0, s[0:1]                            // 00000000188C: DC6A0000 00000001
	s_nop 0                                                    // 000000001894: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001898: BFB60003
	s_endpgm                                                   // 00000000189C: BFB00000
