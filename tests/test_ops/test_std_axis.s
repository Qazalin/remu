
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_5525_45>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s6, s2, s4                                       // 00000000161C: 80060402
	s_addc_u32 s7, s3, s5                                      // 000000001620: 82070503
	s_mov_b64 s[2:3], 0                                        // 000000001624: BE820180
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001628: BF870009
	s_add_u32 s8, s6, s2                                       // 00000000162C: 80080206
	s_addc_u32 s9, s7, s3                                      // 000000001630: 82090307
	s_add_u32 s2, s2, 0x50eec                                  // 000000001634: 8002FF02 00050EEC
	s_clause 0x7                                               // 00000000163C: BF850007
	s_load_b32 s10, s[8:9], null                               // 000000001640: F4000284 F8000000
	s_load_b32 s11, s[8:9], 0x5654                             // 000000001648: F40002C4 F8005654
	s_load_b32 s12, s[8:9], 0xaca8                             // 000000001650: F4000304 F800ACA8
	s_load_b32 s13, s[8:9], 0x102fc                            // 000000001658: F4000344 F80102FC
	s_load_b32 s14, s[8:9], 0x15950                            // 000000001660: F4000384 F8015950
	s_load_b32 s15, s[8:9], 0x1afa4                            // 000000001668: F40003C4 F801AFA4
	s_load_b32 s16, s[8:9], 0x205f8                            // 000000001670: F4000404 F80205F8
	s_load_b32 s17, s[8:9], 0x25c4c                            // 000000001678: F4000444 F8025C4C
	s_addc_u32 s3, s3, 0                                       // 000000001680: 82038003
	s_cmp_eq_u32 s2, 0xf2cc4                                   // 000000001684: BF06FF02 000F2CC4
	s_waitcnt lgkmcnt(0)                                       // 00000000168C: BF89FC07
	v_add_f32_e32 v0, s10, v0                                  // 000000001690: 0600000A
	s_load_b32 s10, s[8:9], 0x2b2a0                            // 000000001694: F4000284 F802B2A0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000169C: BF8700A1
	v_add_f32_e32 v0, s11, v0                                  // 0000000016A0: 0600000B
	s_load_b32 s11, s[8:9], 0x308f4                            // 0000000016A4: F40002C4 F80308F4
	v_add_f32_e32 v0, s12, v0                                  // 0000000016AC: 0600000C
	s_load_b32 s12, s[8:9], 0x35f48                            // 0000000016B0: F4000304 F8035F48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016B8: BF8700A1
	v_add_f32_e32 v0, s13, v0                                  // 0000000016BC: 0600000D
	s_load_b32 s13, s[8:9], 0x3b59c                            // 0000000016C0: F4000344 F803B59C
	v_add_f32_e32 v0, s14, v0                                  // 0000000016C8: 0600000E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016CC: BF870001
	v_add_f32_e32 v0, s15, v0                                  // 0000000016D0: 0600000F
	s_clause 0x2                                               // 0000000016D4: BF850002
	s_load_b32 s14, s[8:9], 0x40bf0                            // 0000000016D8: F4000384 F8040BF0
	s_load_b32 s15, s[8:9], 0x46244                            // 0000000016E0: F40003C4 F8046244
	s_load_b32 s8, s[8:9], 0x4b898                             // 0000000016E8: F4000204 F804B898
	v_add_f32_e32 v0, s16, v0                                  // 0000000016F0: 06000010
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016F4: BF8700A1
	v_add_f32_e32 v0, s17, v0                                  // 0000000016F8: 06000011
	s_waitcnt lgkmcnt(0)                                       // 0000000016FC: BF89FC07
	v_add_f32_e32 v0, s10, v0                                  // 000000001700: 0600000A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001704: BF870091
	v_add_f32_e32 v0, s11, v0                                  // 000000001708: 0600000B
	v_add_f32_e32 v0, s12, v0                                  // 00000000170C: 0600000C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001710: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 000000001714: 0600000D
	v_add_f32_e32 v0, s14, v0                                  // 000000001718: 0600000E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000171C: BF870091
	v_add_f32_e32 v0, s15, v0                                  // 000000001720: 0600000F
	v_add_f32_e32 v0, s8, v0                                   // 000000001724: 06000008
	s_cbranch_scc0 65471                                       // 000000001728: BFA1FFBF <r_5525_45+0x28>
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000172C: BF870001
	v_dual_mul_f32 v0, 0x3cb60b61, v0 :: v_dual_mov_b32 v1, 0  // 000000001730: C8D000FF 00000080 3CB60B61
	s_add_u32 s0, s0, s4                                       // 00000000173C: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001740: 82010501
	global_store_b32 v1, v0, s[0:1]                            // 000000001744: DC6A0000 00000001
	s_nop 0                                                    // 00000000174C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001750: BFB60003
	s_endpgm                                                   // 000000001754: BFB00000
