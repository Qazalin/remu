
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n42>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_max_f32_e64 v0, s2, s2                                   // 000000001638: D5100000 00000402
	v_max_f32_e64 v1, -s2, -s2                                 // 000000001640: D5100001 60000402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001648: BF870091
	v_dual_max_f32 v0, 0, v0 :: v_dual_max_f32 v1, 0, v1       // 00000000164C: CA940080 00000280
	v_add_f32_e32 v0, v0, v1                                   // 000000001654: 06000300
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001658: BF870091
	v_add_f32_e32 v0, 1.0, v0                                  // 00000000165C: 060000F2
	v_div_scale_f32 v1, null, v0, v0, s2                       // 000000001660: D6FC7C01 000A0100
	v_div_scale_f32 v4, vcc_lo, s2, v0, s2                     // 000000001668: D6FC6A04 000A0002
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001670: BF8700B2
	v_rcp_f32_e32 v2, v1                                       // 000000001674: 7E045501
	s_waitcnt_depctr 0xfff                                     // 000000001678: BF880FFF
	v_fma_f32 v3, -v1, v2, 1.0                                 // 00000000167C: D6130003 23CA0501
	v_fmac_f32_e32 v2, v3, v2                                  // 000000001684: 56040503
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001688: BF870091
	v_mul_f32_e32 v3, v4, v2                                   // 00000000168C: 10060504
	v_fma_f32 v5, -v1, v3, v4                                  // 000000001690: D6130005 24120701
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001698: BF870091
	v_fmac_f32_e32 v3, v5, v2                                  // 00000000169C: 56060505
	v_fma_f32 v1, -v1, v3, v4                                  // 0000000016A0: D6130001 24120701
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_div_fmas_f32 v1, v1, v2, v3                              // 0000000016AC: D6370001 040E0501
	v_div_fixup_f32 v0, v1, v0, s2                             // 0000000016B4: D6270000 000A0101
	v_mov_b32_e32 v1, 0                                        // 0000000016BC: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 0000000016C0: DC6A0000 00000001
	s_nop 0                                                    // 0000000016C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016CC: BFB60003
	s_endpgm                                                   // 0000000016D0: BFB00000
